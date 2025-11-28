"""Modal deployment for ABCD Detector."""

import asyncio
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import modal

auth_scheme = HTTPBearer()

app = modal.App("abcd-detector", secrets=[modal.Secret.from_name("abcd-web-auth")])

# Create image with dependencies from requirements.txt
image = (
  modal.Image.debian_slim(python_version="3.11")
  .apt_install("ffmpeg")  # Required for video processing
  .pip_install_from_requirements("requirements.txt")
  .uv_pip_install("fastapi[standard]")
  .add_local_python_source("models")
  .add_local_python_source("annotations_evaluation")
  .add_local_python_source("configuration")
  .add_local_python_source("helpers")
  .add_local_python_source("gcp_api_services")
  .add_local_python_source("evaluation_services")
  .add_local_python_source("features_repository")
  .add_local_python_source("llms_evaluation")
  .add_local_python_source("prompts")
  .add_local_python_source("custom_evaluation")
)


def setup_gcp_credentials():
  """Write GCP service account JSON to temp file and set env var."""
  sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
  if not sa_json:
    raise ValueError(
      "GCP_SERVICE_ACCOUNT_JSON not found. "
      "Create Modal secret: modal secret create gcp-credentials GCP_SERVICE_ACCOUNT_JSON='$(cat key.json)'"
    )

  # Write to temp file
  fd, path = tempfile.mkstemp(suffix=".json")
  with os.fdopen(fd, "w") as f:
    f.write(sa_json)

  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
  return path


@app.function(
  image=image,
  secrets=[modal.Secret.from_name("gcp-credentials")],
  timeout=600,  # 10 min for video processing
)
def assess_video(
  gcs_uri: str,
  project_id: str,
  bucket_name: str = "",
  brand_name: str = "",
  brand_variations: str = "",
  products: str = "",
  products_categories: str = "",
  call_to_actions: str = "",
  use_annotations: bool = False,
  run_long_form_abcd: bool = True,
  run_shorts: bool = True,
  # Additional parameters for full parity with main.py
  project_zone: str = "us-central1",
  use_llms: bool = True,
  extract_brand_metadata: bool | None = None,  # None = auto-detect based on brand_name
  verbose: bool = True,
  creative_provider_type: str = "GCS",  # "GCS" or "YOUTUBE"
  features_to_evaluate: str = "",  # Comma-separated feature IDs
  # Storage/output parameters
  bq_dataset_name: str = "",
  bq_table_name: str = "",
  assessment_file: str = "",
  knowledge_graph_api_key: str = "",
  # LLM configuration
  llm_name: str = "gemini-2.5-pro",
  llm_location: str = "us-central1",
  max_output_tokens: int = 65535,
  temperature: float = 1.0,
  top_p: float = 0.95,
  # Annotation thresholds
  early_time_seconds: float = 5.0,
  confidence_threshold: float = 0.5,
  face_surface_threshold: float = 0.15,
  logo_size_threshold: float = 3.5,
  avg_shot_duration_seconds: float = 2.0,
  dynamic_cutoff_ms: float = 3000.0,
) -> dict:
  """
  Run ABCD assessment on a video.

  Args:
      gcs_uri: Video URI (GCS gs://... or YouTube https://youtube.com/...)
      project_id: GCP project ID
      bucket_name: GCS bucket name (extracted from gcs_uri if not provided)
      brand_name: Brand name for the video
      brand_variations: Comma-separated brand name variations
      products: Comma-separated branded products
      products_categories: Comma-separated product categories
      call_to_actions: Comma-separated call-to-action phrases
      use_annotations: Whether to use Video Intelligence annotations
      run_long_form_abcd: Evaluate long-form ABCD features
      run_shorts: Evaluate YouTube Shorts features
      project_zone: GCP project zone (default: us-central1)
      use_llms: Whether to use LLMs for evaluation (default: True)
      extract_brand_metadata: Extract brand info from video (None=auto based on brand_name)
      verbose: Enable verbose output (default: True)
      creative_provider_type: "GCS" or "YOUTUBE" (default: GCS)
      features_to_evaluate: Comma-separated feature IDs to evaluate (empty=all)
      bq_dataset_name: BigQuery dataset name for storing results
      bq_table_name: BigQuery table name for storing results
      assessment_file: Local file path to write results
      knowledge_graph_api_key: Google Knowledge Graph API key
      llm_name: LLM model name (default: gemini-2.5-pro)
      llm_location: LLM model location (default: us-central1)
      max_output_tokens: Max LLM output tokens (default: 65535)
      temperature: LLM temperature (default: 1.0)
      top_p: LLM top_p parameter (default: 0.95)
      early_time_seconds: Annotation threshold - early appearance (default: 5.0)
      confidence_threshold: Annotation threshold - confidence level (default: 0.5)
      face_surface_threshold: Annotation threshold - face detection (default: 0.15)
      logo_size_threshold: Annotation threshold - logo size (default: 3.5)
      avg_shot_duration_seconds: Annotation threshold - shot duration (default: 2.0)
      dynamic_cutoff_ms: Annotation threshold - max clip length (default: 3000.0)

  Returns:
      Dictionary containing the video assessment results
  """
  # Setup GCP credentials before importing GCP libraries
  creds_path = setup_gcp_credentials()

  try:
    # Import after setting credentials (these use ADC)
    import logging
    import models
    from annotations_evaluation import annotations_generation
    from configuration import Configuration
    from evaluation_services import video_evaluation_service
    from helpers import generic_helpers
    from models import CreativeProviderType, VideoFeatureCategory

    # Determine creative provider type
    provider_type = CreativeProviderType[creative_provider_type.upper()]

    # Validate URI matches provider type
    if provider_type == CreativeProviderType.GCS and "gs://" not in gcs_uri:
      raise ValueError(
        f"Creative provider GCS does not match video URI {gcs_uri}. "
        "Expected gs:// URI."
      )
    if provider_type == CreativeProviderType.YOUTUBE and "youtube.com" not in gcs_uri:
      raise ValueError(
        f"Creative provider YOUTUBE does not match video URI {gcs_uri}. "
        "Expected youtube.com URL."
      )

    # Extract bucket name from URI if not provided (GCS only)
    if not bucket_name and gcs_uri.startswith("gs://"):
      bucket_name = gcs_uri.replace("gs://", "").split("/")[0]

    # Determine extract_brand_metadata setting
    should_extract_brand = (
      extract_brand_metadata if extract_brand_metadata is not None
      else not brand_name  # Auto-detect: extract if no brand_name provided
    )

    # Build configuration
    config = Configuration()
    config.project_id = project_id
    config.project_zone = project_zone
    config.bucket_name = bucket_name
    config.video_uris = [gcs_uri]
    config.use_annotations = use_annotations
    config.use_llms = use_llms
    config.run_long_form_abcd = run_long_form_abcd
    config.run_shorts = run_shorts
    config.creative_provider_type = provider_type
    config.extract_brand_metadata = should_extract_brand
    config.verbose = verbose
    config.knowledge_graph_api_key = knowledge_graph_api_key

    # Storage/output settings
    config.bq_dataset_name = bq_dataset_name
    config.bq_table_name = bq_table_name
    config.assessment_file = assessment_file

    # Feature filtering
    if features_to_evaluate:
      config.features_to_evaluate = [
        f.strip() for f in features_to_evaluate.split(",") if f.strip()
      ]

    # Set LLM parameters
    config.llm_params.model_name = llm_name
    config.llm_params.location = llm_location
    config.llm_params.generation_config = {
      "max_output_tokens": max_output_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "response_schema": {"type": "string"},
    }

    # Set annotation thresholds
    config.early_time_seconds = early_time_seconds
    config.confidence_threshold = confidence_threshold
    config.face_surface_threshold = face_surface_threshold
    config.logo_size_threshold = logo_size_threshold
    config.avg_shot_duration_seconds = avg_shot_duration_seconds
    config.dynamic_cutoff_ms = dynamic_cutoff_ms

    # Set brand details if provided
    if brand_name:
      config.brand_name = brand_name
      config.brand_variations = [
        v.strip() for v in brand_variations.split(",") if v.strip()
      ]
      config.branded_products = [
        p.strip() for p in products.split(",") if p.strip()
      ]
      config.branded_products_categories = [
        c.strip() for c in products_categories.split(",") if c.strip()
      ]
      config.branded_call_to_actions = [
        a.strip() for a in call_to_actions.split(",") if a.strip()
      ]

    # Generate annotations if requested (GCS only)
    if use_annotations and provider_type == CreativeProviderType.GCS:
      annotations_generation.generate_video_annotations(config, gcs_uri)

    # Trim video for long-form ABCD (requires first 5 seconds, GCS only)
    if run_long_form_abcd and provider_type == CreativeProviderType.GCS:
      generic_helpers.trim_video(config, gcs_uri)

    # Run evaluations
    long_form_results = []
    shorts_results = []

    if run_long_form_abcd:
      long_form_results = (
        video_evaluation_service.video_evaluation_service.evaluate_features(
          config=config,
          video_uri=gcs_uri,
          features_category=VideoFeatureCategory.LONG_FORM_ABCD,
        )
      )

    if run_shorts:
      shorts_results = (
        video_evaluation_service.video_evaluation_service.evaluate_features(
          config=config,
          video_uri=gcs_uri,
          features_category=VideoFeatureCategory.SHORTS,
        )
      )

    # Build assessment
    assessment = models.VideoAssessment(
      brand_name=config.brand_name,
      video_uri=gcs_uri,
      long_form_abcd_evaluated_features=long_form_results,
      shorts_evaluated_features=shorts_results,
      config=config,  # Include config for BQ storage
    )

    # Print assessments if verbose
    if verbose:
      if long_form_results:
        generic_helpers.print_abcd_assessment(
          assessment.brand_name, assessment.video_uri, long_form_results
        )
      if shorts_results:
        generic_helpers.print_abcd_assessment(
          assessment.brand_name, assessment.video_uri, shorts_results
        )

    # Store results in BigQuery if configured
    if bq_table_name:
      generic_helpers.store_in_bq(config, assessment)

    # Cleanup local files
    generic_helpers.remove_local_video_files()

    return assessment.to_dict()

  finally:
    # Cleanup credentials file
    if os.path.exists(creds_path):
      os.remove(creds_path)


@app.function(
  image=image,
  secrets=[modal.Secret.from_name("gcp-credentials")],
  timeout=600,
)
@modal.fastapi_endpoint(method="POST")
async def assess_video_endpoint(
  request: dict,
  token: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> dict:
  """
  HTTP endpoint for ABCD video assessment.

  Request body:
  {
      "gcs_uri": "gs://bucket/video.mp4" or "https://youtube.com/...",
      "project_id": "my-gcp-project",
      "bucket_name": "optional-bucket-name",
      "brand_name": "optional brand name",
      "brand_variations": "optional,variations",
      "products": "optional,products",
      "products_categories": "optional,categories",
      "call_to_actions": "optional,ctas",
      "use_annotations": false,
      "run_long_form_abcd": true,
      "run_shorts": true,
      "project_zone": "us-central1",
      "use_llms": true,
      "extract_brand_metadata": null,
      "verbose": true,
      "creative_provider_type": "GCS",
      "features_to_evaluate": "",
      "bq_dataset_name": "",
      "bq_table_name": "",
      "assessment_file": "",
      "knowledge_graph_api_key": "",
      "llm_name": "gemini-2.5-pro",
      "llm_location": "us-central1",
      "max_output_tokens": 65535,
      "temperature": 1.0,
      "top_p": 0.95,
      "early_time_seconds": 5.0,
      "confidence_threshold": 0.5,
      "face_surface_threshold": 0.15,
      "logo_size_threshold": 3.5,
      "avg_shot_duration_seconds": 2.0,
      "dynamic_cutoff_ms": 3000.0
  }

  Headers:
      Authorization: Bearer <token> matching ABCD_ADMIN secret

  Returns:
      Video assessment results as JSON
  """
  # Check authorization token
  if token.credentials != os.environ["AUTH_TOKEN"]:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Incorrect bearer token",
      headers={"WWW-Authenticate": "Bearer"},
    )

  result = assess_video.remote(
    gcs_uri=request["gcs_uri"],
    project_id=request["project_id"],
    bucket_name=request.get("bucket_name", ""),
    brand_name=request.get("brand_name", ""),
    brand_variations=request.get("brand_variations", ""),
    products=request.get("products", ""),
    products_categories=request.get("products_categories", ""),
    call_to_actions=request.get("call_to_actions", ""),
    use_annotations=request.get("use_annotations", False),
    run_long_form_abcd=request.get("run_long_form_abcd", True),
    run_shorts=request.get("run_shorts", True),
    # Additional parameters
    project_zone=request.get("project_zone", "us-central1"),
    use_llms=request.get("use_llms", True),
    extract_brand_metadata=request.get("extract_brand_metadata"),
    verbose=request.get("verbose", True),
    creative_provider_type=request.get("creative_provider_type", "GCS"),
    features_to_evaluate=request.get("features_to_evaluate", ""),
    bq_dataset_name=request.get("bq_dataset_name", ""),
    bq_table_name=request.get("bq_table_name", ""),
    assessment_file=request.get("assessment_file", ""),
    knowledge_graph_api_key=request.get("knowledge_graph_api_key", ""),
    llm_name=request.get("llm_name", "gemini-2.5-pro"),
    llm_location=request.get("llm_location", "us-central1"),
    max_output_tokens=request.get("max_output_tokens", 65535),
    temperature=request.get("temperature", 1.0),
    top_p=request.get("top_p", 0.95),
    early_time_seconds=request.get("early_time_seconds", 5.0),
    confidence_threshold=request.get("confidence_threshold", 0.5),
    face_surface_threshold=request.get("face_surface_threshold", 0.15),
    logo_size_threshold=request.get("logo_size_threshold", 3.5),
    avg_shot_duration_seconds=request.get("avg_shot_duration_seconds", 2.0),
    dynamic_cutoff_ms=request.get("dynamic_cutoff_ms", 3000.0),
  )
  return JSONResponse(content=result)


@app.function(
  image=image,
  secrets=[modal.Secret.from_name("gcp-credentials")],
  timeout=600,
)
@modal.fastapi_endpoint(method="POST")
async def stream(
  request: dict,
  token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
  """
  Streaming HTTP endpoint for ABCD video assessment.

  Works exactly like assess_video_endpoint but uses streaming responses
  to avoid timeout issues. Sends keep-alive messages every 60 seconds
  while processing, then sends the final result.

  The response is newline-delimited JSON (NDJSON):
  - Keep-alive messages: {"status": "processing"}
  - Final result: {"status": "complete", "result": {...}}
  - Error: {"status": "error", "error": "..."}

  Request body: Same as assess_video_endpoint
  Headers: Authorization: Bearer <token>
  """
  # Check authorization token
  if token.credentials != os.environ["AUTH_TOKEN"]:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Incorrect bearer token",
      headers={"WWW-Authenticate": "Bearer"},
    )

  async def stream_evaluation():
    """Generator that yields keep-alive messages while evaluation runs."""
    # Run the blocking Modal function in a thread pool
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    def run_assessment():
      return assess_video.remote(
        gcs_uri=request["gcs_uri"],
        project_id=request["project_id"],
        bucket_name=request.get("bucket_name", ""),
        brand_name=request.get("brand_name", ""),
        brand_variations=request.get("brand_variations", ""),
        products=request.get("products", ""),
        products_categories=request.get("products_categories", ""),
        call_to_actions=request.get("call_to_actions", ""),
        use_annotations=request.get("use_annotations", False),
        run_long_form_abcd=request.get("run_long_form_abcd", True),
        run_shorts=request.get("run_shorts", True),
        project_zone=request.get("project_zone", "us-central1"),
        use_llms=request.get("use_llms", True),
        extract_brand_metadata=request.get("extract_brand_metadata"),
        verbose=request.get("verbose", True),
        creative_provider_type=request.get("creative_provider_type", "GCS"),
        features_to_evaluate=request.get("features_to_evaluate", ""),
        bq_dataset_name=request.get("bq_dataset_name", ""),
        bq_table_name=request.get("bq_table_name", ""),
        assessment_file=request.get("assessment_file", ""),
        knowledge_graph_api_key=request.get("knowledge_graph_api_key", ""),
        llm_name=request.get("llm_name", "gemini-2.5-pro"),
        llm_location=request.get("llm_location", "us-central1"),
        max_output_tokens=request.get("max_output_tokens", 65535),
        temperature=request.get("temperature", 1.0),
        top_p=request.get("top_p", 0.95),
        early_time_seconds=request.get("early_time_seconds", 5.0),
        confidence_threshold=request.get("confidence_threshold", 0.5),
        face_surface_threshold=request.get("face_surface_threshold", 0.15),
        logo_size_threshold=request.get("logo_size_threshold", 3.5),
        avg_shot_duration_seconds=request.get("avg_shot_duration_seconds", 2.0),
        dynamic_cutoff_ms=request.get("dynamic_cutoff_ms", 3000.0),
      )

    # Start the assessment task
    future = loop.run_in_executor(executor, run_assessment)

    # Send initial processing message
    yield json.dumps({"status": "processing"}) + "\n"

    # Send keep-alive messages every 60 seconds while waiting
    while not future.done():
      try:
        # Wait up to 60 seconds for the result
        await asyncio.wait_for(asyncio.shield(future), timeout=60)
        break  # Result is ready
      except asyncio.TimeoutError:
        # Still processing, send keep-alive
        yield json.dumps({"status": "processing"}) + "\n"

    # Get the result (will raise if there was an error)
    try:
      result = await future
      yield json.dumps({"status": "complete", "result": result}) + "\n"
    except Exception as e:
      yield json.dumps({"status": "error", "error": str(e)}) + "\n"
    finally:
      executor.shutdown(wait=False)

  return StreamingResponse(
    stream_evaluation(),
    media_type="application/x-ndjson",
  )


# Local entry point for testing
@app.local_entrypoint()
def main(
  gcs_uri: str,
  project_id: str,
  bucket_name: str = "",
  brand_name: str = "",
  brand_variations: str = "",
  products: str = "",
  products_categories: str = "",
  call_to_actions: str = "",
  use_annotations: bool = False,
  run_long_form_abcd: bool = True,
  run_shorts: bool = True,
  creative_provider_type: str = "GCS",
  use_llms: bool = True,
  extract_brand_metadata: bool = None,
  verbose: bool = True,
  features_to_evaluate: str = "",
  bq_dataset_name: str = "",
  bq_table_name: str = "",
  llm_name: str = "gemini-2.5-pro",
):
  """
  Test the assessment locally.

  Usage:
      modal run abcd-modal.py --gcs-uri "gs://bucket/video.mp4" --project-id "my-project"
      modal run abcd-modal.py --gcs-uri "https://youtube.com/..." --project-id "my-project" --creative-provider-type YOUTUBE
  """
  result = assess_video.remote(
    gcs_uri=gcs_uri,
    project_id=project_id,
    bucket_name=bucket_name,
    brand_name=brand_name,
    brand_variations=brand_variations,
    products=products,
    products_categories=products_categories,
    call_to_actions=call_to_actions,
    use_annotations=use_annotations,
    run_long_form_abcd=run_long_form_abcd,
    run_shorts=run_shorts,
    creative_provider_type=creative_provider_type,
    use_llms=use_llms,
    extract_brand_metadata=extract_brand_metadata,
    verbose=verbose,
    features_to_evaluate=features_to_evaluate,
    bq_dataset_name=bq_dataset_name,
    bq_table_name=bq_table_name,
    llm_name=llm_name,
  )
  import json

  print(json.dumps(result, indent=2))
