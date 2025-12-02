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

                    CLASSIFY THE FOLLOWING:
                    1. SPEAKER GENDER: male, female, non_binary, unidentifiable, none
                    2. SPEAKER ACCENT: american, british, indian, chinese, other, none
                    3. SPEAKER LANGUAGE: EN, ES, RU, other, none

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "gender": "male|female|non_binary|unidentifiable|none",
                        "accent": "american|british|indian|chinese|other|none",
                        "language": "EN|ES|RU|other|none"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Strong Speaker presence (0.8-1.0): Voiceover is primary
                        - Moderate Speaker presence (0.6-0.7): Voiceover is strong
                        - Weak Speaker presence (0.2-0.5): Voiceover is secondary
                        - No Speaker presence (0.0-0.1): Voiceover is non-existent
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # NARRATIVE & EMOTIONAL FEATURES
        # =====================================================================
        VideoFeature(
            id="emotional_arc",
            name="Emotional Arc Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the emotional journey and arc of the advertisement.
                """,
            prompt_template="""
                    Analyze the emotional arc and journey of this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. PRIMARY EMOTIONS: select from [melancholy, nostalgia, catharsis, triumph, anxiety, relief, joy, excitement, fear, sadness, anger, surprise, anticipation, empathy, pride]
                    2. EMOTIONAL INTENSITY: subtle, moderate, intense
                    3. SENTIMENT SHIFT: does emotion change? If yes, from what to what?

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "primary_emotions": ["emotion1", "emotion2"],
                        "emotional_intensity": "subtle|moderate|intense",
                        "sentiment_shift": true|false,
                        "starting_emotion": "emotion or null",
                        "ending_emotion": "emotion or null"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Clear emotional arc (0.8-1.0): Obvious emotional journey
                        - Moderate emotional content (0.6-0.7): Some emotional elements
                        - Subtle emotions (0.3-0.5): Understated emotional content
                        - No clear emotions (0.0-0.2): Purely informational
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="narrative_structure",
            name="Narrative Structure Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the narrative structure and storytelling approach of the ad.
                """,
            prompt_template="""
                    Analyze the narrative structure of this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. HOOK TYPE: provocative_statement, question, challenge, relatable_pain_point, visual_surprise, audio_grab
                    2. PAYOFF TYPE: gameplay_moment, reveal, tutorial, joke_resolution, emotional_resolution, product_showcase
                    3. STORY STRUCTURE: setup_payoff, problem_solution, journey, compilation, vignette, testimonial

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "hook_type": "provocative_statement|question|challenge|relatable_pain_point|visual_surprise|audio_grab",
                        "payoff_type": "gameplay_moment|reveal|tutorial|joke_resolution|emotional_resolution|product_showcase",
                        "story_structure": "setup_payoff|problem_solution|journey|compilation|vignette|testimonial"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Clear narrative (0.8-1.0): Well-defined story structure
                        - Moderate narrative (0.6-0.7): Some narrative elements
                        - Weak narrative (0.3-0.5): Minimal story structure
                        - No narrative (0.0-0.2): Random or purely visual
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # GAMING-SPECIFIC FEATURES
        # =====================================================================
        VideoFeature(
            id="gameplay_presentation",
            name="Gameplay Presentation Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze how gameplay is presented and what type of gaming moment is showcased.
                """,
            prompt_template="""
                    Analyze the gameplay presentation in this gaming advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. GAMEPLAY TYPE: clutch_moment, fail, progression_milestone, satisfying_mechanic, rare_occurrence, competitive_play, casual_play, speedrun, none
                    2. GAMEPLAY EMOTION: triumphant, devastating, satisfying, frustrating, surprising, nostalgic
                    3. SKILL DISPLAY: pro_play, beginner_relatable, lucky_moment, strategy_showcase

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "gameplay_type": "clutch_moment|fail|progression_milestone|satisfying_mechanic|rare_occurrence|competitive_play|casual_play|speedrun|none",
                        "gameplay_emotion": "triumphant|devastating|satisfying|frustrating|surprising|nostalgic",
                        "skill_display": "pro_play|beginner_relatable|lucky_moment|strategy_showcase"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Clear gameplay focus (0.8-1.0): Gameplay is central
                        - Some gameplay (0.6-0.7): Gameplay is present but not focus
                        - Minimal gameplay (0.3-0.5): Brief gameplay shown
                        - No gameplay (0.0-0.2): No actual gameplay footage
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="in_group_signaling",
            name="In-Group Signaling Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze how the ad signals to specific gaming communities and what level of game knowledge it assumes.
                """,
            prompt_template="""
                    Analyze the in-group signaling and community targeting in this ad.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. NICHE REFERENCE LEVEL: casual_player, dedicated_player, hardcore_community
                    2. ASSUMES GAME KNOWLEDGE: true/false, and what types (mechanics, characters, meta, memes, history)
                    3. HAS INSIDE JOKES: true/false

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "niche_reference_level": "casual_player|dedicated_player|hardcore_community",
                        "assumes_game_knowledge": true|false,
                        "knowledge_types": ["mechanics", "characters", "meta", "memes", "history"],
                        "has_inside_jokes": true|false
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Strong in-group signals (0.8-1.0): Heavy community references
                        - Moderate signals (0.6-0.7): Some insider elements
                        - Light signals (0.3-0.5): Minimal community specificity
                        - No signals (0.0-0.2): Completely generic/accessible
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # FORMAT & PRODUCTION FEATURES
        # =====================================================================
        VideoFeature(
            id="text_overlay_strategy",
            name="Text Overlay Strategy Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze how text overlays are used in the advertisement.
                """,
            prompt_template="""
                    Analyze the text overlay strategy in this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. TEXT FUNCTION: hook, punchline, context, commentary, meme_format, subtitles, call_to_action, none
                    2. TEXT TIMING: upfront, throughout, delayed_reveal, intermittent, none
                    3. FONT STYLE: casual_phone, impact_meme, minimalist, game_ui_styled, handwritten, bold_dramatic, none

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "text_function": "hook|punchline|context|commentary|meme_format|subtitles|call_to_action|none",
                        "text_timing": "upfront|throughout|delayed_reveal|intermittent|none",
                        "font_style": "casual_phone|impact_meme|minimalist|game_ui_styled|handwritten|bold_dramatic|none"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Strategic text use (0.8-1.0): Text is crucial to message
                        - Moderate text use (0.6-0.7): Text enhances content
                        - Minimal text (0.3-0.5): Some text present
                        - No text (0.0-0.2): No text overlays
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="audio_strategy",
            name="Audio Strategy Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the audio strategy and sound design of the advertisement.
                """,
            prompt_template="""
                    Analyze the audio strategy in this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. AUDIO TYPE: trending_song, original_audio, game_sound, voiceover, silence, sound_effects, mixed
                    2. AUDIO MODIFICATION: slowed_reverb, sped_up, pitched, lo_fi, original, bass_boosted, remix
                    3. MOOD MATCH: reinforces, contrasts, ironic

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "audio_type": "trending_song|original_audio|game_sound|voiceover|silence|sound_effects|mixed",
                        "audio_modification": "slowed_reverb|sped_up|pitched|lo_fi|original|bass_boosted|remix",
                        "mood_match": "reinforces|contrasts|ironic"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Strategic audio (0.8-1.0): Audio is crucial to impact
                        - Good audio use (0.6-0.7): Audio enhances content
                        - Basic audio (0.3-0.5): Functional audio
                        - Poor/no audio (0.0-0.2): Audio not leveraged
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="visual_style",
            name="Visual Style Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the visual style and production quality of the advertisement.
                """,
            prompt_template="""
                    Analyze the visual style in this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. PRODUCTION QUALITY: screenshot_dump, minimal_edit, moderate_edit, polished, heavily_edited
                    2. PACING: slow_burn, rapid_cut, single_shot, build_up, varied
                    3. ASPECT RATIO: vertical_native, horizontal_letterbox, square_centered, mixed
                    4. VISUAL EFFECTS: select any that apply [zoom_ins, shake_effects, color_grading, filters, transitions, none]

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "production_quality": "screenshot_dump|minimal_edit|moderate_edit|polished|heavily_edited",
                        "pacing": "slow_burn|rapid_cut|single_shot|build_up|varied",
                        "aspect_ratio": "vertical_native|horizontal_letterbox|square_centered|mixed",
                        "visual_effects": ["zoom_ins", "shake_effects", "color_grading", "filters", "transitions", "none"]
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Distinctive style (0.8-1.0): Clear visual identity
                        - Good visual execution (0.6-0.7): Competent visuals
                        - Basic visuals (0.3-0.5): Functional but plain
                        - Poor visuals (0.0-0.2): Low quality
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # PLATFORM OPTIMIZATION FEATURES
        # =====================================================================
        VideoFeature(
            id="engagement_mechanics",
            name="Engagement Mechanics Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the engagement optimization techniques used in the advertisement.
                """,
            prompt_template="""
                    Analyze the engagement mechanics in this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. SCROLL STOPPER: text_hook, visual_surprise, audio_grab, movement, face_presence, bright_colors, none
                    2. COMMENT BAIT: true/false
                    3. DUET/STITCH POTENTIAL: true/false
                    4. ALGORITHM SIGNALS: select any that apply [watch_time_optimized, rewatch_potential, share_worthy, save_worthy, none]

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "scroll_stopper": "text_hook|visual_surprise|audio_grab|movement|face_presence|bright_colors|none",
                        "comment_bait": true|false,
                        "duet_stitch_potential": true|false,
                        "algorithm_signals": ["watch_time_optimized", "rewatch_potential", "share_worthy", "save_worthy"]
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Strong engagement design (0.8-1.0): Multiple techniques
                        - Good engagement (0.6-0.7): Some optimization
                        - Basic engagement (0.3-0.5): Minimal optimization
                        - No engagement focus (0.0-0.2): Not optimized
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # META-CONTENT FEATURES
        # =====================================================================
        VideoFeature(
            id="self_awareness_level",
            name="Self-Awareness Level Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the level of self-awareness and irony in the advertisement.
                """,
            prompt_template="""
                    Analyze the self-awareness and irony level in this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. IRONY PRESENT: true/false
                    2. META COMMENTARY: select any that apply [gaming_culture, advertising, platform, fourth_wall, none]
                    3. SINCERITY LEVEL: genuine, ironic_genuine, fully_ironic, absurdist

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "irony_present": true|false,
                        "meta_commentary": ["gaming_culture", "advertising", "platform", "fourth_wall"],
                        "sincerity_level": "genuine|ironic_genuine|fully_ironic|absurdist"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Clear meta-awareness (0.8-1.0): Obvious irony/commentary
                        - Some awareness (0.6-0.7): Subtle meta elements
                        - Minimal awareness (0.3-0.5): Mostly straightforward
                        - No meta elements (0.0-0.2): Completely sincere
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="cultural_positioning",
            name="Cultural Positioning Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the cultural positioning and demographic targeting of the advertisement.
                """,
            prompt_template="""
                    Analyze the cultural positioning in this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. TARGET SUBCULTURE: gaming_gen_z, mobile_casual, competitive_gamers, nostalgia_millennials, content_creators, general_audience
                    2. GENDER FRAMING: male_targeted, female_targeted, gender_neutral, subverts_stereotypes
                    3. GENERATIONAL MARKERS: select any that apply [music_era, meme_format, cultural_reference, communication_style, none]

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "target_subculture": "gaming_gen_z|mobile_casual|competitive_gamers|nostalgia_millennials|content_creators|general_audience",
                        "gender_framing": "male_targeted|female_targeted|gender_neutral|subverts_stereotypes",
                        "generational_markers": ["music_era", "meme_format", "cultural_reference", "communication_style"]
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Clear targeting (0.8-1.0): Obvious demographic focus
                        - Moderate targeting (0.6-0.7): Some demographic signals
                        - Light targeting (0.3-0.5): Minimal specificity
                        - No targeting (0.0-0.2): Completely generic
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # VIRALITY POTENTIAL FEATURES
        # =====================================================================
        VideoFeature(
            id="remixability",
            name="Remixability Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the remixability and template potential of the advertisement format.
                """,
            prompt_template="""
                    Analyze the remixability potential of this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. TEMPLATE POTENTIAL: high, medium, low, none
                    2. CAPTION FLEXIBILITY: true/false
                    3. CROSS-GAME APPLICABLE: true/false
                    4. UGC POTENTIAL: high, medium, low, none

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "template_potential": "high|medium|low|none",
                        "caption_flexibility": true|false,
                        "cross_game_applicable": true|false,
                        "ugc_potential": "high|medium|low|none"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Highly remixable (0.8-1.0): Easy to recreate/adapt
                        - Moderately remixable (0.6-0.7): Some adaptability
                        - Limited remixability (0.3-0.5): Difficult to recreate
                        - Not remixable (0.0-0.2): Unique/one-off content
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="social_currency",
            name="Social Currency Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the social currency and shareability value of the advertisement.
                """,
            prompt_template="""
                    Analyze the social currency potential of this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. GATEKEEPING VALUE: true/false
                    2. RELATABILITY SCOPE: universal, gamers_broadly, game_specific, ultra_niche
                    3. CONVERSATION STARTER: true/false
                    4. IDENTITY EXPRESSION: true/false

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "gatekeeping_value": true|false,
                        "relatability_scope": "universal|gamers_broadly|game_specific|ultra_niche",
                        "conversation_starter": true|false,
                        "identity_expression": true|false
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - High social currency (0.8-1.0): Strong share motivation
                        - Moderate currency (0.6-0.7): Some share value
                        - Low currency (0.3-0.5): Limited share appeal
                        - No currency (0.0-0.2): No social value
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # QUALITY METRICS FEATURES
        # =====================================================================
        VideoFeature(
            id="completion_rate_prediction",
            name="Completion Rate Prediction",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Predict the likely completion rate based on content structure and pacing.
                """,
            prompt_template="""
                    Analyze and predict the completion rate potential of this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. HOOK STRENGTH: strong, moderate, weak
                    2. PAYOFF TIMING: early, middle, late, none
                    3. DROP-OFF RISK: low, medium, high
                    4. REWATCH POTENTIAL: true/false
                    5. PREDICTED COMPLETION: high, moderate, low

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "hook_strength": "strong|moderate|weak",
                        "payoff_timing": "early|middle|late|none",
                        "drop_off_risk": "low|medium|high",
                        "rewatch_potential": true|false,
                        "predicted_completion": "high|moderate|low"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - High completion likely (0.8-1.0): Strong hook + good pacing
                        - Moderate completion (0.6-0.7): Decent structure
                        - Low completion risk (0.3-0.5): Some issues
                        - Poor completion likely (0.0-0.2): Major problems
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        VideoFeature(
            id="brand_integration",
            name="Brand Integration Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze how well the brand/product is integrated into the content.
                """,
            prompt_template="""
                    Analyze the brand integration in this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    CLASSIFY THE FOLLOWING:
                    1. INTEGRATION LEVEL: seamless, obvious_but_accepted, jarring, hidden
                    2. CALL TO ACTION: implicit, soft, direct, aggressive, none
                    3. NATIVE FEEL: true/false

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "integration_level": "seamless|obvious_but_accepted|jarring|hidden",
                        "call_to_action": "implicit|soft|direct|aggressive|none",
                        "native_feel": true|false
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Excellent integration (0.8-1.0): Seamless brand presence
                        - Good integration (0.6-0.7): Acceptable ad format
                        - Poor integration (0.3-0.5): Feels like an ad
                        - Bad integration (0.0-0.2): Jarring/off-putting
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
        # =====================================================================
        # SPECIAL COMBINATION FEATURE
        # =====================================================================
        VideoFeature(
            id="hook_payoff_structure",
            name="Hook-Payoff Structure Analysis",
            category=VideoFeatureCategory.CUSTOM,
            sub_category=VideoFeatureSubCategory.NONE,
            video_segment=VideoSegment.FULL_VIDEO,
            evaluation_criteria="""
                    Analyze the hook-payoff structure, particularly universal_hook_niche_payoff patterns.
                """,
            prompt_template="""
                    Analyze the hook-payoff structure of this advertisement.

                    VIDEO METADATA:
                    {metadata_summary}

                    This analysis looks for powerful advertising patterns like:
                    - Universal hook + niche payoff (broad appeal opening, specific satisfaction)
                    - Niche hook + universal payoff (insider opening, relatable satisfaction)

                    CLASSIFY THE FOLLOWING:
                    1. HOOK UNIVERSALITY: universal, semi_universal, niche, ultra_niche
                    2. PAYOFF SPECIFICITY: universal, semi_universal, niche, ultra_niche
                    3. STRUCTURE PATTERN: universal_hook_niche_payoff, niche_hook_universal_payoff, consistent_universal, consistent_niche, inverted
                    4. EFFECTIVENESS: highly_effective, effective, neutral, ineffective

                    Return your classification in the 'evaluation' field as a JSON object:
                    {{
                        "hook_universality": "universal|semi_universal|niche|ultra_niche",
                        "payoff_specificity": "universal|semi_universal|niche|ultra_niche",
                        "structure_pattern": "universal_hook_niche_payoff|niche_hook_universal_payoff|consistent_universal|consistent_niche|inverted",
                        "effectiveness": "highly_effective|effective|neutral|ineffective"
                    }}

                    CONFIDENCE CONSIDERATIONS:
                        - Clear pattern (0.8-1.0): Obvious structure choice
                        - Moderate clarity (0.6-0.7): Some structure visible
                        - Unclear structure (0.3-0.5): Hard to identify
                        - No pattern (0.0-0.2): Random/no structure
                """,
            extra_instructions=[],
            evaluation_method=EvaluationMethod.LLMS_CATEGORICAL,
            evaluation_function="",
            include_in_evaluation=True,
            group_by=VideoSegment.FULL_VIDEO,
        ),
    ]

    return feature_configs
