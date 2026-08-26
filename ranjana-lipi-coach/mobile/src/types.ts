export type PracticeMode = "app_suggested" | "free_practice" | "assessment";

export type Character = {
  id: number;
  name: string;
  display_label: string;
  region_grid_rows: number;
  region_grid_cols: number;
};

export type User = {
  id: number;
  email: string;
  display_name: string | null;
  created_at: string;
};

export type TokenResponse = {
  access_token: string;
  token_type: string;
};

export type RegionScore = {
  row: number;
  col: number;
  region?: string;
  label?: string;
  score?: number;
  error?: number;
  mean_error?: number;
  normalized_error?: number;
  normalized_score?: number;
  z_score?: number;
  missing_ratio?: number;
  extra_ratio?: number;
  dominant_issue?: "missing" | "extra" | "none";
  message?: string;
  is_problem?: boolean;
  insufficient_data?: boolean;
};

export type RegionFeedback = {
  target_class?: string;
  predicted_class?: string;
  recognizer_confidence?: number;
  overall_score: number;
  insufficient_input?: boolean;
  wrong_character?: boolean;
  feedback_method?: string;
  template_stats?: Record<string, unknown>;
  autoencoder_overall_score?: number;
  message?: string | null;
  ink_pixel_count?: number;
  min_required_ink_pixels?: number;
  warning?: string | null;
  problem_regions?: RegionScore[];
  all_regions?: RegionScore[];
  fine_grid?: {
    problem_regions?: RegionScore[];
    all_regions?: RegionScore[];
  };
  broad_bands?: {
    problem_regions?: RegionScore[];
    all_regions?: RegionScore[];
  };
  regions?: RegionScore[];
  region_scores?: RegionScore[];
  [key: string]: unknown;
};

export type Attempt = {
  id: number;
  user_id: number;
  character_id: number;
  mode: PracticeMode;
  image_path: string;
  overall_score: number | null;
  region_feedback: RegionFeedback | null;
  created_at: string;
};

export type PracticeAttemptResponse = {
  attempt: Attempt;
  overall_score: number;
  region_feedback: RegionFeedback;
};

export type Progress = {
  id: number;
  user_id: number;
  character_id: number;
  attempts_count: number;
  best_score: number | null;
  last_practiced_at: string | null;
  mastered: boolean;
};

export type ProgressDashboardItem = {
  character: Character;
  progress: Progress | null;
};

export type CharacterProgressDetail = {
  character: Character;
  progress: Progress | null;
  attempts: Attempt[];
};

export type PracticeHeatmapDay = {
  date: string;
  attempts_count: number;
  average_score: number | null;
  best_score: number | null;
};

export type UserProfileStats = {
  total_attempts: number;
  practiced_characters: number;
  mastered_characters: number;
  average_score: number | null;
  best_score: number | null;
  current_streak_days: number;
  longest_streak_days: number;
};

export type UserProfile = {
  user: User;
  stats: UserProfileStats;
  heatmap: PracticeHeatmapDay[];
  generated_at: string;
};

export type SelectedImage = {
  uri: string;
  name: string;
  type: string;
  source: "camera" | "gallery" | "canvas" | "demo_canvas";
};
