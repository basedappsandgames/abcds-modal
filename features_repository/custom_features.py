#!/usr/bin/env python3

###########################################################################
#
#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###########################################################################

"""Module with the supported ABCD feature configurations for Custom"""

from models import (
    EvaluationMethod,
    VideoFeature,
    VideoFeatureCategory,
    VideoFeatureSubCategory,
    VideoSegment,
)


def get_custom_feature_configs() -> list[VideoFeature]:
    """Gets all the supported ABCD/Shorts features
    Returns:
    feature_configs: list of feature configurations
    """
    feature_configs = [
        VideoFeature(
            id="speaker_gender_accent_language",
            name="Speaker Gender Accent Language",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze whether the video uses voice over or a speaker with a distinct accent or gender, then list that gender and accent.
                """,
            prompt_template="""
                    Analyze the speaker and/or voice-over of this short-form video.

                    VIDEO METADATA:
                    {metadata_summary}

                    ANALYZE FOR TWO DISTINCT SPEAKER QUALITIES:
                    1. SPEAKER GENDER:
                        a) Male
                            - Deeper voice
                        b) Female
                            - Higher voice
                        c) Non-binary/unidentifiable
                    2. SPEAKER ACCENT:
                        a) American
                        b) UK / British / English
                        c) Other:
                            - Indian
                            - Singapore / Chinese
                    3. SPEAKER LANGUAGE:
                        a) English
                        b) Spanish
                        c) Russian
                        d) etc

                    FORMAT RESPONSE AS JSON:
                    {{
                        "detected": boolean,  # TRUE for has speaker, FALSE for no speaker or voice-over
                        "confidence_score": float,
                        "evaluation": {{
                            "gender": str, # Male, female, non-binary/unidentifiable,
                            "accent": str, # American, UK
                            "language": str, # EN, ES, etc
                        }}
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Strong Speaker presence (0.8-1.0): Voiceover is primary
                        - Moderate Speaker presence (0.6-0.7): Voiceover is strong
                        - Weak Speaker presence (0.2-0.5): Voiceover is secondary
                        - No Speaker presence (0.0-0.1): Voiceover is non-existent

                    IMPORTANT NOTES:
                        1. Speakers may be of different ages
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,  # confirm
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
    ]

    return feature_configs
