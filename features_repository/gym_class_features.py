#!/usr/bin/env python3

"""Module with the supported feature configurations for Gym Class VR content"""

from models import (
    EvaluationMethod,
    VideoFeature,
    VideoFeatureCategory,
    VideoFeatureSubCategory,
    VideoSegment,
)


def get_gym_class_feature_configs() -> list[VideoFeature]:
    """Gets all the supported Gym Class quality features.

    These features are designed so that ALL features being True = Quality Score 5.
    Each True flag represents a positive quality indicator.

    Returns:
        feature_configs: list of feature configurations
    """
    feature_configs = [
        # =====================================================================
        # TECHNICAL QUALITY FEATURES
        # =====================================================================
        VideoFeature(
            id="gc_high_video_quality",
            name="High Video Quality",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video has high video quality with good resolution,
                clarity, and visual fidelity. Poor quality includes blurry, pixelated,
                or low-resolution footage.
            """,
            prompt_template="""
                Analyze the video quality of this Gym Class short-form video.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Is the video resolution high (720p or better)?
                - Is the footage clear and not blurry or pixelated?
                - Does the capture quality appear to be good fidelity (Quest 3 level preferred)?
                - Are visuals crisp and well-defined?

                Return TRUE if the video has HIGH video quality.
                Return FALSE if the video has poor quality, is blurry, pixelated, or low resolution.

                CONFIDENCE CONSIDERATIONS:
                    - High quality (0.8-1.0): Clear, crisp, high-resolution footage
                    - Moderate quality (0.5-0.7): Acceptable but not ideal quality
                    - Poor quality (0.0-0.4): Blurry, pixelated, low resolution
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="gc_smooth_frame_rate",
            name="Smooth Frame Rate",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video has a smooth, consistent frame rate without
                stuttering, juddering, or choppy playback.
            """,
            prompt_template="""
                Analyze the frame rate smoothness of this Gym Class short-form video.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Does the video play smoothly without stuttering?
                - Is the frame rate consistent throughout?
                - Are there any choppy or juddering moments?
                - Does motion appear fluid?

                Return TRUE if the video has SMOOTH frame rate.
                Return FALSE if the video stutters, judders, or has inconsistent/choppy playback.

                CONFIDENCE CONSIDERATIONS:
                    - Smooth (0.8-1.0): Fluid, consistent frame rate throughout
                    - Acceptable (0.5-0.7): Minor frame rate issues but mostly smooth
                    - Choppy (0.0-0.4): Noticeable stuttering or frame drops
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # VR CAPTURE QUALITY FEATURES
        # =====================================================================
        VideoFeature(
            id="gc_no_black_borders",
            name="No Black Borders from VR Capture",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video fills the entire screen without black borders
                or black squares on the sides that commonly appear from VR capture.
            """,
            prompt_template="""
                Analyze whether this Gym Class video has black borders from VR capture.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Does the video fill the entire screen/phone screen?
                - Are there any black squares or borders on the sides?
                - Is the aspect ratio properly optimized for short-form vertical video?
                - Are there any black letterboxing areas?

                Return TRUE if the video fills the ENTIRE screen with NO black borders.
                Return FALSE if there are black borders, black squares, or the video doesn't fill the screen.

                CONFIDENCE CONSIDERATIONS:
                    - Full screen (0.8-1.0): Video completely fills the frame
                    - Minor borders (0.4-0.7): Small black areas present
                    - Significant borders (0.0-0.3): Large black borders or squares visible
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="gc_no_boundary_mesh",
            name="No Boundary Mesh Visible",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the VR boundary/guardian mesh (blue mesh around player
                that indicates play space limits) is visible at any point in the video.
            """,
            prompt_template="""
                Analyze whether this Gym Class video shows the VR boundary/guardian mesh.

                VIDEO METADATA:
                {metadata_summary}

                The VR boundary mesh is a blue/teal grid or mesh pattern that appears
                in VR when players approach the edge of their play space. It's a safety
                feature that shouldn't appear in polished content.

                EVALUATE:
                - Is there any blue mesh or grid visible at any point?
                - Does the boundary/guardian visualization appear anywhere?
                - Are there any VR play space limit indicators visible?

                Return TRUE if there is NO boundary mesh visible anywhere in the video.
                Return FALSE if the blue boundary mesh appears at any point.

                CONFIDENCE CONSIDERATIONS:
                    - No mesh (0.8-1.0): Clean footage without any boundary indicators
                    - Brief mesh (0.4-0.7): Mesh briefly visible
                    - Visible mesh (0.0-0.3): Mesh clearly visible in parts of the video
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # AUDIO FEATURES
        # =====================================================================
        VideoFeature(
            id="gc_balanced_audio",
            name="Balanced Audio Levels",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the audio levels are balanced and pleasant, with
                appropriate volume levels for background music and any speaking.
            """,
            prompt_template="""
                Analyze the audio balance of this Gym Class short-form video.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Are the audio levels balanced (not too loud, not too quiet)?
                - Is the song choice pleasant and appropriate?
                - Is the background music at an appropriate level?
                - Is the audio grating, unpleasant, or poorly balanced?

                Return TRUE if audio levels and song choice are BALANCED and PLEASANT.
                Return FALSE if audio is unbalanced, too loud, too quiet, or grating.

                CONFIDENCE CONSIDERATIONS:
                    - Well balanced (0.8-1.0): Pleasant, professional audio mix
                    - Acceptable (0.5-0.7): Minor balance issues
                    - Poor balance (0.0-0.4): Unbalanced, grating, or unpleasant audio
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="gc_clear_audio_separation",
            name="Clear Audio Separation",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if speaking and background audio are easy to parse and
                differentiate from each other.
            """,
            prompt_template="""
                Analyze the audio separation in this Gym Class short-form video.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - If there is speaking, is it easy to hear and understand?
                - Can you clearly differentiate between speech and background audio/music?
                - Does background audio/music overpower any speaking?
                - Is the audio mix clean and well-separated?

                Return TRUE if speaking and background audio are EASY to parse/differentiate.
                Return FALSE if audio elements are muddled, hard to differentiate, or speech is drowned out.
                Return TRUE if there is no speaking (only background audio that is clear).

                CONFIDENCE CONSIDERATIONS:
                    - Clear separation (0.8-1.0): Easy to distinguish all audio elements
                    - Acceptable (0.5-0.7): Some overlap but understandable
                    - Poor separation (0.0-0.4): Difficult to parse different audio sources
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # PERSPECTIVE & HOOK FEATURES
        # =====================================================================
        VideoFeature(
            id="gc_first_person_start",
            name="Starts in First Person Perspective",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FIRST_5_SECS_VIDEO,
            evaluation_criteria="""
                Determine if the video starts in first person perspective (POV)
                rather than third person perspective.
            """,
            prompt_template="""
                Analyze the opening perspective of this Gym Class short-form video.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Does the video start from a first person (POV) perspective?
                - Is the initial view showing the player's viewpoint as if you are the player?
                - Or does it start in third person, showing the player's avatar from outside?

                Return TRUE if the video starts in FIRST PERSON (1st person/POV) perspective.
                Return FALSE if the video starts in third person perspective.

                CONFIDENCE CONSIDERATIONS:
                    - Clear 1st person (0.8-1.0): Obviously POV perspective at start
                    - Ambiguous (0.4-0.6): Hard to determine perspective
                    - Clear 3rd person (0.0-0.3): Obviously third person at start
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FIRST_5_SECS_VIDEO,
        ),
        VideoFeature(
            id="gc_early_movement_hook",
            name="Movement and Hook in First Two Seconds",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FIRST_5_SECS_VIDEO,
            evaluation_criteria="""
                Determine if the video includes movement and a hook during the
                first two seconds to capture viewer attention immediately.
            """,
            prompt_template="""
                Analyze the first two seconds of this Gym Class short-form video.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Is there movement/action within the first 2 seconds?
                - Is there a hook to capture attention immediately?
                - Does something interesting happen right away?
                - Does the video start with engaging content vs. static/boring opening?

                Return TRUE if there is MOVEMENT and a HOOK in the first 2 seconds.
                Return FALSE if the opening is static, slow, or lacks immediate engagement.

                CONFIDENCE CONSIDERATIONS:
                    - Strong hook (0.8-1.0): Immediate movement and attention-grabbing content
                    - Moderate hook (0.5-0.7): Some movement but weak hook
                    - No hook (0.0-0.4): Static or slow start
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FIRST_5_SECS_VIDEO,
        ),
        VideoFeature(
            id="gc_compelling_hook",
            name="Has Compelling Hook",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video has a compelling hook that makes viewers
                want to watch longer. This can include interesting text, interesting
                sounds, NBA/pro basketball clips, an interesting shot, or prompting
                text like "Why was he so mad?", "Toxic Kid 1v1", etc.
            """,
            prompt_template="""
                Analyze whether this Gym Class video has a compelling hook.

                VIDEO METADATA:
                {metadata_summary}

                A hook can include:
                - Interesting text prompts ("Why was he so mad?", "Toxic Kid 1v1", "Famous NBA player in G-League")
                - Interesting sounds or audio
                - Clips from NBA or other pro basketball leagues
                - An interesting or surprising shot
                - Text that prompts users to watch longer
                - Visual elements that create curiosity

                EVALUATE:
                - Does the video have text that creates curiosity or prompts continued viewing?
                - Are there interesting sounds or audio hooks?
                - Does it utilize pro basketball clips or references?
                - Is there an interesting or surprising shot?
                - Does it make you want to keep watching?

                Return TRUE if the video has a COMPELLING HOOK.
                Return FALSE if the video lacks any hook or attention-grabbing elements.

                CONFIDENCE CONSIDERATIONS:
                    - Strong hook (0.8-1.0): Clear hook that drives engagement
                    - Moderate hook (0.5-0.7): Some hook elements present
                    - No hook (0.0-0.4): Lacks compelling hook elements
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # LENGTH & DURATION FEATURES
        # =====================================================================
        VideoFeature(
            id="gc_ideal_length",
            name="Ideal Video Length (9-20 seconds)",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video length is within the ideal range of 9-20 seconds
                for short-form content.
            """,
            prompt_template="""
                Analyze the length of this Gym Class short-form video.

                VIDEO METADATA:
                {metadata_summary}

                The ideal length for Gym Class short-form content is 9-20 seconds.
                - Too short (<9 seconds): May not have enough content
                - Ideal (9-20 seconds): Optimal for engagement
                - Too long (>20 seconds, especially >60 seconds): May lose viewers

                EVALUATE:
                - What is the video duration?
                - Is it within the 9-20 second ideal range?
                - Is it excessively long (60+ seconds)?

                Return TRUE if the video is around 9-20 SECONDS (ideal length).
                Return FALSE if the video is too short (<9s) or too long (>30s).

                CONFIDENCE CONSIDERATIONS:
                    - Ideal length (0.8-1.0): 9-20 seconds
                    - Acceptable length (0.5-0.7): 6-30 seconds
                    - Poor length (0.0-0.4): Under 6 seconds or over 60 seconds
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.ANNOTATIONS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="gc_meets_minimum_duration",
            name="Meets Minimum Duration (6+ seconds)",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video meets the minimum duration requirement
                of at least 6 seconds.
            """,
            prompt_template="""
                Verify this Gym Class video meets minimum duration requirements.

                VIDEO METADATA:
                {metadata_summary}

                Videos under 6 seconds are considered too short and automatically
                receive a low quality score.

                EVALUATE:
                - Is the video at least 6 seconds long?

                Return TRUE if the video is AT LEAST 6 SECONDS long.
                Return FALSE if the video is under 6 seconds.

                CONFIDENCE CONSIDERATIONS:
                    - Meets minimum (0.9-1.0): 6+ seconds
                    - Below minimum (0.0-0.3): Under 6 seconds
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.ANNOTATIONS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # CONTENT RELEVANCE FEATURES
        # =====================================================================
        VideoFeature(
            id="gc_broad_relevance",
            name="Has Broad Relevance Beyond Gym Class Community",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video clip has relevance and appeal to viewers
                outside of the Gym Class community, not just existing players.
            """,
            prompt_template="""
                Analyze whether this Gym Class video has broad appeal.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Would someone who doesn't play Gym Class find this interesting?
                - Does the content have universal appeal (exciting plays, humor, skill)?
                - Is this just a "New Update in Gym Class" type video only for existing players?
                - Could this attract new viewers to the game?
                - Is this niche content only understandable to existing players?

                Return TRUE if the clip has BROAD RELEVANCE to viewers outside Gym Class community.
                Return FALSE if the content only appeals to existing Gym Class players.

                CONFIDENCE CONSIDERATIONS:
                    - Broad appeal (0.8-1.0): Anyone would find this entertaining
                    - Moderate appeal (0.5-0.7): Some broader appeal
                    - Niche appeal (0.0-0.4): Only appeals to existing community
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="gc_sufficient_gameplay_footage",
            name="Sufficient Gym Class Footage (50%+)",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if at least half of the video contains actual Gym Class
                gameplay footage (not screenshots, external clips, or unrelated content).
                Gym Class is a 3D, first-person VR game with bright colors, where players play basketball, football, baseball, boxing and other sports.
            """,
            prompt_template="""
                Analyze the Gym Class footage proportion in this video.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - What percentage of the video is actual Gym Class gameplay?
                - Is at least half (50%+) of the footage from Gym Class?
                - Does it avoid being mostly screenshots, Discord images, or external content?
                - Is there actual gameplay vs. just static images?

                Return TRUE if AT LEAST HALF (50%+) of the video is Gym Class footage.
                Return FALSE if less than half is Gym Class footage, or if it's mostly screenshots/external content.

                CONFIDENCE CONSIDERATIONS:
                    - Mostly gameplay (0.8-1.0): 75%+ Gym Class footage
                    - Half gameplay (0.5-0.7): 50-75% Gym Class footage
                    - Low gameplay (0.0-0.4): Under 50% Gym Class footage
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # EDITING & PRODUCTION FEATURES
        # =====================================================================
        VideoFeature(
            id="gc_has_editing",
            name="Has Video Editing",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video has been edited rather than being just
                raw uploaded gameplay with no editing.
            """,
            prompt_template="""
                Analyze whether this Gym Class video has been edited.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Is this more than just raw uploaded gameplay?
                - Are there any edits, cuts, transitions, or effects?
                - Does it include text overlays, music additions, or other edits?
                - Is it just a compilation of shots with no editing?
                - Has any post-production work been done?

                Return TRUE if the video HAS EDITING (cuts, effects, text, music, transitions).
                Return FALSE if it's just raw uploaded gameplay with no editing.

                CONFIDENCE CONSIDERATIONS:
                    - Well edited (0.8-1.0): Clear editing, multiple effects/cuts
                    - Some editing (0.5-0.7): Basic editing present
                    - No editing (0.0-0.4): Raw gameplay, no post-production
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="gc_no_capcut_logo",
            name="No CapCut Logo",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video has a CapCut watermark/logo visible,
                which indicates amateur editing and reduces quality score.
            """,
            prompt_template="""
                Check if this Gym Class video has a CapCut logo/watermark.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Is there a "CapCut" logo or watermark visible anywhere in the video?
                - Is there a CapCut outro/ending card?
                - Are there any editing software watermarks?

                Return TRUE if there is NO CapCut logo or watermark.
                Return FALSE if a CapCut logo or watermark is visible.

                CONFIDENCE CONSIDERATIONS:
                    - No watermark (0.9-1.0): Clean video without branding
                    - Watermark present (0.0-0.3): CapCut or other editing watermark visible
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="gc_appropriate_text_coverage",
            name="Appropriate Text Coverage",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if text overlays don't take up too much room (less than
                2/3 of the screen) and don't obfuscate the gameplay.
            """,
            prompt_template="""
                Analyze the text coverage in this Gym Class video.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - How much screen space do text overlays occupy?
                - Does text take up more than 2/3 of the screen?
                - Does text obfuscate or cover important gameplay?
                - Is text appropriately sized and positioned?

                Return TRUE if text coverage is APPROPRIATE (less than 2/3 of screen, doesn't obfuscate gameplay).
                Return FALSE if text takes up too much room (2/3 or more) or obfuscates gameplay.
                Return TRUE if there is no text (no text = no coverage issue).

                CONFIDENCE CONSIDERATIONS:
                    - Appropriate text (0.8-1.0): Text is well-sized and positioned
                    - Moderate coverage (0.5-0.7): Some obstruction but acceptable
                    - Excessive coverage (0.0-0.4): Text covers too much or obfuscates content
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # CONTENT QUALITY FEATURES
        # =====================================================================
        VideoFeature(
            id="gc_is_understandable",
            name="Video is Understandable",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video clearly conveys what it's trying to show
                and is easy to understand vs. being confusing or unclear.
            """,
            prompt_template="""
                Analyze whether this Gym Class video is easy to understand.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Is it clear what the video is trying to convey or show?
                - Can you understand the purpose/point of the video?
                - Is it confusing or hard to follow?
                - Does it make sense to viewers?

                Return TRUE if the video is UNDERSTANDABLE and clearly conveys its message.
                Return FALSE if it's hard to understand what the video is trying to convey.

                CONFIDENCE CONSIDERATIONS:
                    - Clear message (0.8-1.0): Obvious purpose and easy to follow
                    - Somewhat clear (0.5-0.7): Message is understandable with effort
                    - Confusing (0.0-0.4): Hard to understand point or purpose
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="gc_not_low_effort_content",
            name="Not Low-Effort Content",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video is NOT low-effort content like "I got CC",
                "Help me get CC", Discord screenshots, or username screenshots.
            """,
            prompt_template="""
                Check if this Gym Class video is low-effort content.

                VIDEO METADATA:
                {metadata_summary}

                LOW-EFFORT CONTENT INCLUDES:
                - "I got CC" celebration videos
                - "Help me get CC" request videos
                - Screenshots of usernames in Discord announcements
                - Just Discord screenshots
                - Static image slideshows with no gameplay

                EVALUATE:
                - Is this a "I got CC" or "Help me get CC" type video?
                - Is this primarily screenshots from Discord?
                - Is this just showing usernames or announcements?
                - Does this have actual gameplay content?

                Return TRUE if this is NOT low-effort content (actual quality gameplay/content).
                Return FALSE if this is low-effort content (CC videos, Discord screenshots, etc.).

                CONFIDENCE CONSIDERATIONS:
                    - Quality content (0.8-1.0): Actual gameplay/creative content
                    - Mixed content (0.5-0.7): Some low-effort elements but also gameplay
                    - Low-effort (0.0-0.4): Primarily low-effort content types
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="gc_not_just_compilation",
            name="Not Just a Compilation with No Context",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the video is NOT just a random compilation of shots
                with no text, context, or narrative thread.
            """,
            prompt_template="""
                Analyze whether this Gym Class video is a contextless compilation.

                VIDEO METADATA:
                {metadata_summary}

                EVALUATE:
                - Is this just random shots strung together with no context?
                - Is there any text, narrative, or theme tying the content together?
                - Does the video have a point or purpose?
                - Is there any structure or storytelling?

                Return TRUE if the video has CONTEXT (text, narrative, theme, or clear purpose).
                Return FALSE if it's just a compilation of random shots with no context.

                CONFIDENCE CONSIDERATIONS:
                    - Has context (0.8-1.0): Clear purpose, text, or narrative thread
                    - Some context (0.5-0.7): Loose theme or minimal context
                    - No context (0.0-0.4): Random compilation with no purpose
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # CONTENT STANDARD FEATURES
        # =====================================================================
        VideoFeature(
            id="content_standard_age_appropriate",
            name="Age Appropriate Content",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the content is age appropriate, in good taste, and free
                of inappropriate language, bigotry, racism, discrimination, or
                inappropriate sexual content.
            """,
            prompt_template="""
                Analyze whether this Gym Class video contains age-appropriate content.

                VIDEO METADATA:
                {metadata_summary}

                CONTENT MUST BE FREE OF:
                - Inappropriate language (profanity, slurs, vulgar language)
                - Bigotry or hate speech
                - Racism or racial discrimination
                - Discrimination based on gender, religion, nationality, disability,
                  sexual orientation, or age
                - Inappropriate sexual gestures or comments
                - Content that is not in good taste

                EDGE CASES:
                - Incidental and limited instances of PG-13 content, eg "shit" in the background music lyrics, are okay
                - Bleeped and censored profanities are okay

                EVALUATE:
                - Is the language appropriate for all ages?
                - Is the content free of discriminatory remarks or themes?
                - Are there any inappropriate sexual gestures or comments?
                - Is the overall tone appropriate and in good taste?

                Return TRUE if the content IS age appropriate and in good taste.
                Return FALSE if there is ANY inappropriate language, discrimination, or sexual content.

                CONFIDENCE CONSIDERATIONS:
                    - Fully appropriate (0.8-1.0): Clean, family-friendly content
                    - Mostly appropriate (0.5-0.7): Minor concerns but generally acceptable
                    - Inappropriate (0.0-0.4): Contains inappropriate content
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="content_standard_community_guidelines",
            name="Adheres to Community Standards",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the content adheres to community standards and does not
                glorify harmful behaviors such as eating disorders, bullying, hate
                crimes, racism, substance abuse, violence, or suicide.
            """,
            prompt_template="""
                Analyze whether this Gym Class video adheres to community standards.

                VIDEO METADATA:
                {metadata_summary}

                CONTENT THAT VIOLATES COMMUNITY STANDARDS (automatic disqualification):
                - Glorifying eating disorders
                - Glorifying or promoting bullying
                - Glorifying hate crimes
                - Glorifying racism or racist behavior
                - Glorifying substance abuse (drugs, alcohol)
                - Glorifying violence or violent behavior
                - Glorifying or promoting suicide or self-harm

                EVALUATE:
                - Does the video glorify or promote any harmful behaviors?
                - Does it depict bullying in a positive light?
                - Does it contain hate crime imagery or promotion?
                - Does it show substance abuse positively?
                - Does it glorify violence beyond normal gameplay?
                - Does it reference or promote self-harm or suicide?

                Return TRUE if the content ADHERES to community standards (no harmful glorification).
                Return FALSE if any community standard violations are present.

                CONFIDENCE CONSIDERATIONS:
                    - Fully compliant (0.8-1.0): No harmful content, positive messaging
                    - Compliant (0.5-0.7): No violations but some edge cases
                    - Violates standards (0.0-0.4): Contains glorification of harmful behaviors
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="content_standard_relevance",
            name="Relevant Gym Class Content",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the content is relevant to Gym Class, has a generally
                positive attitude, and does not negatively compare Gym Class to
                other games or applications.
            """,
            prompt_template="""
                Analyze whether this video has relevant, positive Gym Class content.

                VIDEO METADATA:
                {metadata_summary}

                CONTENT REQUIREMENTS:
                - Must be within or about Gym Class game
                - Should have a generally positive attitude toward the game
                - Should NOT negatively compare Gym Class to other games
                - Should NOT portray negativity toward other games or applications

                EVALUATE:
                - Is the content actually about or within Gym Class?
                - Does the creator have a positive attitude about the game?
                - Does the video compare Gym Class negatively to other VR games?
                - Does the video trash-talk or disparage other games/applications?
                - Is the overall tone positive and supportive of the game?

                Return TRUE if content is RELEVANT and POSITIVE (about Gym Class, positive tone, no negative comparisons).
                Return FALSE if content is off-topic, negative toward Gym Class, or disparages other games.

                CONFIDENCE CONSIDERATIONS:
                    - Highly relevant & positive (0.8-1.0): Clearly about Gym Class with enthusiasm
                    - Relevant (0.5-0.7): About Gym Class but neutral tone
                    - Not relevant or negative (0.0-0.4): Off-topic or negative comparisons
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="content_standard_parsability",
            name="Visually Parsable Content",
            category=VideoFeatureCategory.GYM_CLASS,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                Determine if the content is visually readable and easy to watch,
                without excessive text, emojis, or effects that make it difficult
                to parse or understand.
            """,
            prompt_template="""
                Analyze whether this Gym Class video is visually parsable and easy to watch.

                VIDEO METADATA:
                {metadata_summary}

                PARSABILITY REQUIREMENTS:
                - Content should be visually readable
                - Should NOT have excessive amounts of text cluttering the screen
                - Should NOT have overwhelming emoji usage
                - Should NOT have excessive visual effects that obscure content
                - Viewer should be able to easily follow what's happening

                EVALUATE:
                - Is the video visually clear and easy to follow?
                - Is there too much text on screen at once?
                - Are there excessive emojis cluttering the visuals?
                - Do visual effects (flashes, transitions, filters) make it hard to watch?
                - Can you easily see and understand the gameplay/content?

                Return TRUE if content is VISUALLY PARSABLE (easy to watch and understand).
                Return FALSE if excessive text, emojis, or effects make it difficult to watch.

                CONFIDENCE CONSIDERATIONS:
                    - Highly parsable (0.8-1.0): Clean, easy to watch, well-balanced visuals
                    - Parsable (0.5-0.7): Some visual clutter but watchable
                    - Hard to parse (0.0-0.4): Excessive text, emojis, or effects obscure content
            """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
    ]

    return feature_configs
