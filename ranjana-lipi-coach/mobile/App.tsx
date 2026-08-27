import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import * as SecureStore from "expo-secure-store";
import { LinearGradient } from "expo-linear-gradient";
import Constants from "expo-constants";
import {
  GoogleSignin,
  statusCodes,
} from "@react-native-google-signin/google-signin";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

import {
  DEFAULT_API_BASE_URL,
  changeCurrentUserPassword,
  deactivateCurrentUser,
  fetchAttemptHistory,
  fetchCharacterProgress,
  fetchCharacters,
  fetchCurrentUser,
  fetchProfile,
  fetchPracticeRecommendations,
  fetchProgress,
  loginUser,
  loginWithGoogle,
  registerUser,
  submitPracticeAttempt,
  updateCurrentUser,
} from "./src/api";
import {
  DrawingCanvas,
  type DrawingCanvasHandle,
} from "./src/components/DrawingCanvas";
import { RegionGrid } from "./src/components/RegionGrid";
import type {
  Attempt,
  Character,
  CharacterProgressDetail,
  PracticeAttemptResponse,
  PracticeRecommendation,
  PracticeMode,
  ProgressDashboardItem,
  RegionFeedback,
  SelectedImage,
  User,
  UserProfile,
} from "./src/types";
import {
  GOOGLE_AUTH_CONFIG,
  isGoogleAuthConfigured,
} from "./src/googleAuthConfig";

type Screen =
  | "auth"
  | "home"
  | "practice"
  | "results"
  | "progress"
  | "profile"
  | "history"
  | "character_detail";
type AuthMode = "login" | "register";
type InputMode = "gallery" | "camera" | "canvas";
type SuggestedPick = {
  item: ProgressDashboardItem;
  reason: string;
};

const TOKEN_KEY = "ranjana_lipi_token";
const API_BASE_URL_KEY = "ranjana_lipi_api_base_url";
const STALE_API_BASE_URLS = new Set([
  "http://17.1.112.44:8000",
  "http://192.168.6.155:8000",
]);
const VALIDATED_CLASSES = new Set(["aa", "a", "ka", "da", "dda"]);
const HIDDEN_CHARACTER_NAMES = new Set(["rii", "lu", "luu"]);
const APP_LOGO = require("./assets/app-logo-transparent.png");
const GOOGLE_WEB_CLIENT_ID =
  GOOGLE_AUTH_CONFIG.webClientId || GOOGLE_AUTH_CONFIG.expoClientId;
const GOOGLE_IOS_CLIENT_ID = GOOGLE_AUTH_CONFIG.iosClientId;
const PRACTICE_MODES: { value: PracticeMode; label: string }[] = [
  { value: "app_suggested", label: "Suggestive Learning" },
  { value: "free_practice", label: "Free Practice" },
];
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const DEVANAGARI_LABELS: Record<string, string> = {
  a: "अ",
  aa: "आ",
  ah: "अः",
  ai: "ऐ",
  am: "अं",
  au: "औ",
  ba: "ब",
  bha: "भ",
  ca: "च",
  cha: "छ",
  da: "ड",
  dda: "द",
  ddha: "ध",
  dha: "ढ",
  e: "ए",
  eight: "८",
  five: "५",
  four: "४",
  ga: "ग",
  gha: "घ",
  gyan: "ज्ञ",
  ha: "ह",
  i: "इ",
  ii: "ई",
  ja: "ज",
  jha: "झ",
  ka: "क",
  kha: "ख",
  ksa: "क्ष",
  la: "ल",
  lu: "ऌ",
  luu: "ॡ",
  ma: "म",
  na: "ङ",
  nine: "९",
  nna: "ण",
  nnna: "न",
  nya: "ञ",
  o: "ओ",
  one: "१",
  pa: "प",
  pha: "फ",
  ra: "र",
  ri: "ऋ",
  rii: "ॠ",
  sa: "ष",
  saa: "स",
  seven: "७",
  sha: "श",
  six: "६",
  ta: "ट",
  tha: "ठ",
  three: "३",
  tra: "त्र",
  tta: "त",
  ttha: "थ",
  two: "२",
  u: "उ",
  uu: "ऊ",
  wo: "व",
  ya: "य",
  zero: "०",
};

type CharacterSectionKey = "vowels" | "consonants" | "numbers" | "other";

const CHARACTER_SECTION_LABELS: Record<CharacterSectionKey, string> = {
  vowels: "Vowels",
  consonants: "Consonants",
  numbers: "Numbers",
  other: "Other",
};

const CHARACTER_SECTION_ORDER: CharacterSectionKey[] = [
  "vowels",
  "consonants",
  "numbers",
  "other",
];

const VOWEL_ORDER = [
  "a",
  "aa",
  "i",
  "ii",
  "u",
  "uu",
  "ri",
  "rii",
  "lu",
  "luu",
  "e",
  "ai",
  "o",
  "au",
  "am",
  "ah",
];

const CONSONANT_ORDER = [
  "ka",
  "kha",
  "ga",
  "gha",
  "na",
  "ca",
  "cha",
  "ja",
  "jha",
  "nya",
  "ta",
  "tha",
  "da",
  "dha",
  "nna",
  "tta",
  "ttha",
  "dda",
  "ddha",
  "nnna",
  "pa",
  "pha",
  "ba",
  "bha",
  "ma",
  "ya",
  "ra",
  "la",
  "wo",
  "sha",
  "sa",
  "saa",
  "ha",
  "ksa",
  "tra",
  "gyan",
];

const NUMBER_ORDER = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
];

const CHARACTER_ORDER_MAP = new Map<string, number>(
  [
    ...VOWEL_ORDER.map((name, index) => [name, index] as const),
    ...CONSONANT_ORDER.map((name, index) => [name, 100 + index] as const),
    ...NUMBER_ORDER.map((name, index) => [name, 300 + index] as const),
  ],
);

const THEME = {
  ink: "#393D3F",
  surface: "#FDFDFF",
  muted: "#C6C5B9",
  accent: "#62929E",
  slate: "#546A7B",
  background: "#F5F6F4",
  softAccent: "#E7F0F2",
  softMuted: "#EFEFEB",
  danger: "#B33B2E",
  warning: "#A46A16",
  success: "#2F7D5A",
};

function scoreColor(score: number | null | undefined): string {
  if (typeof score !== "number") {
    return THEME.slate;
  }
  if (score >= 90) {
    return THEME.success;
  }
  if (score >= 70) {
    return THEME.warning;
  }
  return THEME.danger;
}

function characterDisplayLabel(character: Character): string {
  return DEVANAGARI_LABELS[character.name] ?? character.display_label;
}

function characterSection(characterName: string): CharacterSectionKey {
  if (VOWEL_ORDER.includes(characterName)) {
    return "vowels";
  }
  if (CONSONANT_ORDER.includes(characterName)) {
    return "consonants";
  }
  if (NUMBER_ORDER.includes(characterName)) {
    return "numbers";
  }
  return "other";
}

function compareCharacters(a: Character, b: Character): number {
  const aOrder = CHARACTER_ORDER_MAP.get(a.name) ?? 999;
  const bOrder = CHARACTER_ORDER_MAP.get(b.name) ?? 999;
  if (aOrder !== bOrder) {
    return aOrder - bOrder;
  }
  return characterDisplayLabel(a).localeCompare(characterDisplayLabel(b));
}

function groupCharacters(characters: Character[]): Array<{
  key: CharacterSectionKey;
  title: string;
  characters: Character[];
}> {
  return CHARACTER_SECTION_ORDER.map((key) => ({
    key,
    title: CHARACTER_SECTION_LABELS[key],
    characters: characters
      .filter((character) => characterSection(character.name) === key)
      .sort(compareCharacters),
  })).filter((section) => section.characters.length > 0);
}

function characterGlyphUri(baseUrl: string, characterName: string): string {
  return `${baseUrl}/display_glyphs/${characterName}.png?v=structure-good-15`;
}

function demoCanvasUri(baseUrl: string, characterName: string): string {
  return `${baseUrl}/demo_canvas/${characterName}_canvas_demo_high_score.png`;
}

function heatmapColor(attempts: number): string {
  if (attempts <= 0) {
    return THEME.softMuted;
  }
  if (attempts === 1) {
    return "#C8DBDE";
  }
  if (attempts === 2) {
    return "#8DB8C0";
  }
  if (attempts <= 4) {
    return THEME.accent;
  }
  return THEME.ink;
}

function scoreText(score: number | null | undefined): string {
  return typeof score === "number" ? `${score.toFixed(1)}%` : "--";
}

function attemptImageUri(baseUrl: string, attempt: Attempt): string {
  return `${baseUrl}/${attempt.image_path}`;
}

function formatAttemptDate(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function attemptCharacterLabel(
  characters: Character[],
  attempt: Attempt,
): string {
  const character = characters.find((item) => item.id === attempt.character_id);
  return character
    ? characterDisplayLabel(character)
    : `Character ${attempt.character_id}`;
}

function topRegionSummary(feedback: RegionFeedback | null | undefined): string {
  if (feedback?.wrong_character) {
    return "Wrong character";
  }
  if (feedback?.invalid_input) {
    return "Invalid input";
  }
  if (feedback?.insufficient_input) {
    return "Insufficient input";
  }
  const broad = feedback?.broad_bands as
    | { problem_regions?: Array<{ region?: string }> }
    | undefined;
  const fine = feedback?.fine_grid as
    | { problem_regions?: Array<{ region?: string }> }
    | undefined;
  const region =
    broad?.problem_regions?.[0]?.region ?? fine?.problem_regions?.[0]?.region;
  return region ? `Needs work: ${region}` : "No strong region flagged";
}

function hoursSince(timestamp: string | null | undefined, now: number): number {
  if (!timestamp) {
    return Number.POSITIVE_INFINITY;
  }
  return Math.max(0, (now - new Date(timestamp).getTime()) / (1000 * 60 * 60));
}

function ankiStyleIntervalHours(item: ProgressDashboardItem): number {
  const progress = item.progress;
  if (
    !progress ||
    progress.attempts_count === 0 ||
    progress.best_score === null
  ) {
    return 0;
  }

  const attempts = Math.max(1, progress.attempts_count);
  const score = progress.best_score;
  if (score < 70) {
    return 0.25;
  }
  if (score < 85) {
    return Math.min(24, 4 * attempts);
  }
  if (score < 95) {
    return Math.min(24 * 7, 24 * Math.pow(1.7, attempts - 1));
  }
  return Math.min(24 * 30, 24 * 3 * Math.pow(2.3, attempts - 1));
}

function suggestedReason(
  item: ProgressDashboardItem,
  intervalHours: number,
  elapsedHours: number,
): string {
  const progress = item.progress;
  if (!progress || progress.attempts_count === 0) {
    return "New character";
  }
  if ((progress.best_score ?? 0) < 70) {
    return "Needs quick review";
  }
  if (elapsedHours >= intervalHours) {
    return "Due for review";
  }
  return "Weakest upcoming review";
}

function chooseSuggestedPick(
  progress: ProgressDashboardItem[],
): SuggestedPick | null {
  if (progress.length === 0) {
    return null;
  }

  const now = Date.now();
  const ranked = progress
    .map((item, index) => {
      const intervalHours = ankiStyleIntervalHours(item);
      const elapsedHours = hoursSince(item.progress?.last_practiced_at, now);
      const score = item.progress?.best_score ?? 0;
      const dueRatio = intervalHours === 0 ? 10 : elapsedHours / intervalHours;
      const weaknessBoost = Math.max(0, 100 - score) / 25;
      const newBoost = item.progress ? 0 : 4;
      const masteredPenalty = item.progress?.mastered ? 1.5 : 0;
      return {
        item,
        index,
        intervalHours,
        elapsedHours,
        priority: dueRatio + weaknessBoost + newBoost - masteredPenalty,
      };
    })
    .sort((a, b) => b.priority - a.priority || a.index - b.index);

  const top = ranked[0];
  return {
    item: top.item,
    reason: suggestedReason(top.item, top.intervalHours, top.elapsedHours),
  };
}

function toSelectedImage(
  asset: ImagePicker.ImagePickerAsset,
  source: "camera" | "gallery",
): SelectedImage {
  const fileName = asset.fileName ?? `${source}_${Date.now()}.jpg`;
  const extension = fileName.split(".").pop()?.toLowerCase();
  const type = extension === "png" ? "image/png" : "image/jpeg";

  return {
    uri: asset.uri,
    name: fileName,
    type,
    source,
  };
}

function problemRegionText(feedback: RegionFeedback | null): string {
  if (feedback?.wrong_character) {
    return (
      feedback.message ??
      feedback.warning ??
      "This attempt does not match the selected character."
    );
  }

  if (feedback?.insufficient_input) {
    return (
      feedback.message ??
      feedback.warning ??
      "Insufficient input — please draw the full character."
    );
  }

  const regions = feedback?.problem_regions;
  if (!Array.isArray(regions) || regions.length === 0) {
    return "No strong flawed region was flagged.";
  }

  return regions
    .map((region) => {
      const label =
        region.label ?? `row ${region.row + 1}, col ${region.col + 1}`;
      return region.message ? `${label}: ${region.message}` : label;
    })
    .join("\n");
}

function resultStatusText(
  score: number | null | undefined,
  feedback: RegionFeedback | null,
): string {
  if (feedback?.wrong_character) {
    return "Wrong character";
  }
  if (feedback?.invalid_input) {
    return "Invalid input";
  }
  if (feedback?.insufficient_input) {
    return "Draw more clearly";
  }
  if (typeof score !== "number") {
    return "Feedback ready";
  }
  if (score >= 90) {
    return "Great work";
  }
  if (score >= 75) {
    return "Good attempt";
  }
  return "Needs practice";
}

function learnerFeedbackText(feedback: RegionFeedback | null): string {
  if (!feedback) {
    return "Submit an attempt to see handwriting feedback.";
  }
  if (feedback.wrong_character) {
    return (
      feedback.warning ??
      feedback.message ??
      "This looks like a different character. Try writing the selected character again."
    );
  }
  if (feedback.invalid_input) {
    return (
      feedback.warning ??
      feedback.message ??
      "This does not look like a supported character. Try a clear character image."
    );
  }
  if (feedback.insufficient_input) {
    return (
      feedback.message ??
      "Please draw the full character before submitting."
    );
  }

  const broadProblems = feedback.broad_bands?.problem_regions ?? [];
  const mainProblem = broadProblems[0];
  if (mainProblem?.region) {
    return `${mainProblem.region[0].toUpperCase()}${mainProblem.region.slice(
      1,
    )} region needs improvement.`;
  }
  return "No major problem region detected. Keep practicing this form.";
}

function regionDisplayName(region: string): string {
  return `${region[0].toUpperCase()}${region.slice(1)}`;
}

function broadProblemRegions(feedback: RegionFeedback | null): string[] {
  const regions = feedback?.broad_bands?.problem_regions ?? [];
  return regions
    .map((region) => region.region)
    .filter((region): region is string => typeof region === "string");
}

function regionSummary(feedback: RegionFeedback | null): {
  main: string;
  secondary: string;
  clear: string;
} {
  if (feedback?.wrong_character) {
    return {
      main: "Character mismatch",
      secondary: "Check selected character",
      clear: "Score blocked",
    };
  }
  if (feedback?.invalid_input) {
    return {
      main: "Invalid input",
      secondary: "Use a clear character",
      clear: "Try again",
    };
  }
  if (feedback?.insufficient_input) {
    return {
      main: "Input incomplete",
      secondary: "Draw the full form",
      clear: "Try again",
    };
  }

  const problems = broadProblemRegions(feedback);
  const allBands = ["top", "middle", "bottom"];
  return {
    main: problems[0] ? regionDisplayName(problems[0]) : "None",
    secondary: problems
      .slice(1, 3)
      .map(regionDisplayName)
      .join(", ") || "None",
    clear:
      allBands
        .filter((band) => !problems.includes(band))
        .map(regionDisplayName)
        .join(", ") || "Keep practicing",
  };
}

function coachingSuggestion(feedback: RegionFeedback | null): string {
  if (feedback?.wrong_character) {
    return "Choose the correct target character and submit that form again.";
  }
  if (feedback?.invalid_input) {
    return "Use a clear photo or drawing of one supported character, without extra objects or background clutter.";
  }
  if (feedback?.insufficient_input) {
    return "Draw the full character with enough visible strokes before submitting.";
  }

  const main = broadProblemRegions(feedback)[0];
  if (main === "top") {
    return "Focus on completing the upper headline and top stroke clearly.";
  }
  if (main === "middle") {
    return "Focus on keeping the main body and middle loop connected.";
  }
  if (main === "bottom") {
    return "Focus on completing the lower tail or base stroke.";
  }
  return "This attempt looks clean. Repeat it once more to build consistency.";
}

function technicalResultText(feedback: RegionFeedback | null): string {
  if (feedback?.wrong_character) {
    return "The recognizer checks whether the submitted writing matches the selected character before regional scoring is shown.";
  }
  if (feedback?.invalid_input) {
    return "The app first checks whether the upload resembles one of the supported character structures. If no character structure matches strongly enough, scoring is blocked.";
  }
  if (feedback?.feedback_method === "structural_part_mask") {
    return "The app checks whether the submitted writing covers the required structural parts of the selected character.";
  }
  if (feedback?.feedback_method === "statistical_template") {
    return "The app compares the submitted writing with the learned variation range for the selected character.";
  }
  return "The app compares the submitted writing with the expected character structure and highlights the region that needs the most attention.";
}

export default function App() {
  const [screen, setScreen] = useState<Screen>("auth");
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [progress, setProgress] = useState<ProgressDashboardItem[]>([]);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [attemptHistory, setAttemptHistory] = useState<Attempt[]>([]);
  const [characterDetail, setCharacterDetail] =
    useState<CharacterProgressDetail | null>(null);
  const [selectedCharacterId, setSelectedCharacterId] = useState<number | null>(
    null,
  );
  const [selectedMode, setSelectedMode] =
    useState<PracticeMode>("app_suggested");
  const [inputMode, setInputMode] = useState<InputMode>("gallery");
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(
    null,
  );
  const [submittedImage, setSubmittedImage] = useState<SelectedImage | null>(
    null,
  );
  const [result, setResult] = useState<PracticeAttemptResponse | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [accountEmail, setAccountEmail] = useState("");
  const [accountDisplayName, setAccountDisplayName] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [characterSearch, setCharacterSearch] = useState("");
  const [characterPickerOpen, setCharacterPickerOpen] = useState(false);
  const [suggestedReasonText, setSuggestedReasonText] = useState<string | null>(
    null,
  );
  const [suggestedRecommendation, setSuggestedRecommendation] =
    useState<PracticeRecommendation | null>(null);
  const [resultDetailsOpen, setResultDetailsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const drawingRef = useRef<DrawingCanvasHandle>(null);
  const googleConfigured = isGoogleAuthConfigured();
  const googleConfiguredForPlatform =
    Platform.OS === "android"
      ? GOOGLE_WEB_CLIENT_ID.trim().length > 0
      : Platform.OS === "ios"
        ? GOOGLE_AUTH_CONFIG.iosClientId.trim().length > 0
        : googleConfigured;

  const selectedCharacter = useMemo(
    () =>
      characters.find((character) => character.id === selectedCharacterId) ??
      characters[0] ??
      null,
    [characters, selectedCharacterId],
  );

  const filteredCharacters = useMemo(() => {
    const query = characterSearch.trim().toLowerCase();
    const visibleCharacters = characters.filter(
      (character) => !HIDDEN_CHARACTER_NAMES.has(character.name),
    );
    const source = query
      ? visibleCharacters.filter((character) => {
      return (
        character.name.toLowerCase().includes(query) ||
        character.display_label.toLowerCase().includes(query) ||
        characterDisplayLabel(character).includes(query)
      );
    })
      : visibleCharacters;
    return [...source].sort(compareCharacters);
  }, [characterSearch, characters]);

  const filteredCharacterSections = useMemo(
    () => groupCharacters(filteredCharacters),
    [filteredCharacters],
  );

  const progressSections = useMemo(() => {
    return CHARACTER_SECTION_ORDER.map((key) => ({
      key,
      title: CHARACTER_SECTION_LABELS[key],
      items: progress
        .filter((item) => !HIDDEN_CHARACTER_NAMES.has(item.character.name))
        .filter((item) => characterSection(item.character.name) === key)
        .sort((a, b) => compareCharacters(a.character, b.character)),
    })).filter((section) => section.items.length > 0);
  }, [progress]);

  const loadSession = useCallback(async () => {
    const storedBaseUrl = await SecureStore.getItemAsync(API_BASE_URL_KEY);
    const storedToken = await SecureStore.getItemAsync(TOKEN_KEY);
    const resolvedBaseUrl =
      storedBaseUrl && !STALE_API_BASE_URLS.has(storedBaseUrl)
        ? storedBaseUrl
        : DEFAULT_API_BASE_URL;

    if (storedBaseUrl !== resolvedBaseUrl) {
      await SecureStore.setItemAsync(API_BASE_URL_KEY, resolvedBaseUrl);
    }
    setApiBaseUrl(resolvedBaseUrl);

    if (!storedToken) {
      return;
    }

    try {
      const currentUser = await fetchCurrentUser(resolvedBaseUrl, storedToken);
      setToken(storedToken);
      setUser(currentUser);
      setScreen("home");
    } catch {
      await SecureStore.deleteItemAsync(TOKEN_KEY);
    }
  }, []);

  const refreshAppData = useCallback(async () => {
    if (!token) {
      return;
    }

    const [nextCharacters, nextProgress, nextProfile, nextAttempts] =
      await Promise.all([
        fetchCharacters(apiBaseUrl, token),
        fetchProgress(apiBaseUrl, token),
        fetchProfile(apiBaseUrl, token),
        fetchAttemptHistory(apiBaseUrl, token, 50),
      ]);
    setCharacters(nextCharacters);
    setProgress(nextProgress);
    setProfile(nextProfile);
    setAttemptHistory(nextAttempts);
    setSelectedCharacterId(
      (current) => current ?? nextCharacters[0]?.id ?? null,
    );
  }, [apiBaseUrl, token]);

  useEffect(() => {
    void loadSession();
  }, [loadSession]);

  useEffect(() => {
    if (user) {
      setAccountEmail(user.email);
      setAccountDisplayName(user.display_name ?? "");
    }
  }, [user]);

  useEffect(() => {
    if (!googleConfigured) {
      return;
    }
    GoogleSignin.configure({
      iosClientId: GOOGLE_IOS_CLIENT_ID || undefined,
      offlineAccess: false,
      profileImageSize: 120,
      webClientId: GOOGLE_WEB_CLIENT_ID,
    });
  }, [googleConfigured]);

  useEffect(() => {
    if (token) {
      void refreshAppData().catch((error: unknown) => {
        setMessage(
          error instanceof Error ? error.message : "Could not load app data.",
        );
      });
    }
  }, [refreshAppData, token]);

  async function handleGoogleAuth() {
    if (!googleConfiguredForPlatform) {
      setMessage("Google sign-in is not configured for this device yet.");
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      await GoogleSignin.hasPlayServices({
        showPlayServicesUpdateDialog: true,
      });
      await GoogleSignin.signOut().catch(() => undefined);
      const googleUser = await GoogleSignin.signIn();
      const idToken = googleUser.data?.idToken;
      if (!idToken) {
        throw new Error("Google did not return an identity token.");
      }

      const response = await loginWithGoogle(apiBaseUrl, idToken);
      await SecureStore.setItemAsync(TOKEN_KEY, response.access_token);
      await SecureStore.setItemAsync(API_BASE_URL_KEY, apiBaseUrl);
      const currentUser = await fetchCurrentUser(
        apiBaseUrl,
        response.access_token,
      );
      setToken(response.access_token);
      setUser(currentUser);
      setScreen("home");
    } catch (error) {
      const code =
        typeof error === "object" && error !== null && "code" in error
          ? String((error as { code?: unknown }).code)
          : "";
      if (code === statusCodes.SIGN_IN_CANCELLED) {
        setMessage("Google sign-in was cancelled.");
      } else if (code === statusCodes.IN_PROGRESS) {
        setMessage("Google sign-in is already in progress.");
      } else if (code === statusCodes.PLAY_SERVICES_NOT_AVAILABLE) {
        setMessage("Google Play Services is not available or needs updating.");
      } else {
        setMessage(
          error instanceof Error ? error.message : "Google sign-in failed.",
        );
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleAuth() {
    const emailValue = email.trim();
    const displayNameValue =
      displayName.trim() || emailValue.split("@")[0] || "Learner";

    if (!emailValue || !password.trim()) {
      setMessage("Email and password are required.");
      return;
    }

    if (!EMAIL_PATTERN.test(emailValue)) {
      setMessage("Enter a valid email address, for example name@example.com.");
      return;
    }

    if (authMode === "register" && password.length < 8) {
      setMessage("Password must be at least 8 characters.");
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      const response =
        authMode === "register"
          ? await registerUser(
              apiBaseUrl,
              emailValue,
              password,
              displayNameValue,
            )
          : await loginUser(apiBaseUrl, emailValue, password);
      await SecureStore.setItemAsync(TOKEN_KEY, response.access_token);
      await SecureStore.setItemAsync(API_BASE_URL_KEY, apiBaseUrl);
      const currentUser = await fetchCurrentUser(
        apiBaseUrl,
        response.access_token,
      );
      setToken(response.access_token);
      setUser(currentUser);
      setScreen("home");
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Authentication failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleLogout() {
    await SecureStore.deleteItemAsync(TOKEN_KEY);
    setToken(null);
    setUser(null);
    setProfile(null);
    setCurrentPassword("");
    setNewPassword("");
    setScreen("auth");
    setResult(null);
    setSelectedImage(null);
    setSubmittedImage(null);
  }

  async function handleAccountUpdate() {
    if (!token) {
      return;
    }

    const nextEmail = accountEmail.trim();
    const nextDisplayName = accountDisplayName.trim();
    if (!EMAIL_PATTERN.test(nextEmail)) {
      setMessage("Enter a valid email address, for example name@example.com.");
      return;
    }
    if (!nextDisplayName) {
      setMessage("Display name cannot be empty.");
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      const updatedUser = await updateCurrentUser(apiBaseUrl, token, {
        email: nextEmail,
        display_name: nextDisplayName,
      });
      setUser(updatedUser);
      setProfile((current) =>
        current ? { ...current, user: updatedUser } : current,
      );
      setMessage("Account details updated.");
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Could not update account.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function handlePasswordChange() {
    if (!token) {
      return;
    }
    if (!currentPassword || !newPassword) {
      setMessage("Enter your current password and a new password.");
      return;
    }
    if (newPassword.length < 8) {
      setMessage("New password must be at least 8 characters.");
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      await changeCurrentUserPassword(
        apiBaseUrl,
        token,
        currentPassword,
        newPassword,
      );
      setCurrentPassword("");
      setNewPassword("");
      setMessage("Password changed.");
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Could not change password.",
      );
    } finally {
      setLoading(false);
    }
  }

  function confirmDeactivateAccount() {
    Alert.alert(
      "Deactivate account?",
      "You will be logged out and this account will no longer be able to sign in.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Deactivate",
          style: "destructive",
          onPress: () => void handleDeactivateAccount(),
        },
      ],
    );
  }

  async function handleDeactivateAccount() {
    if (!token) {
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      await deactivateCurrentUser(apiBaseUrl, token);
      await handleLogout();
      setMessage("Your account has been deactivated.");
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Could not deactivate account.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function pickFromGallery() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert(
        "Permission needed",
        "Gallery access is needed to choose a handwriting sample.",
      );
      return;
    }

    const image = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: false,
      mediaTypes: ["images"],
      quality: 1,
    });

    if (!image.canceled && image.assets[0]) {
      setSelectedImage(toSelectedImage(image.assets[0], "gallery"));
      setInputMode("gallery");
    }
  }

  async function takePhoto() {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert(
        "Permission needed",
        "Camera access is needed to photograph a handwriting sample.",
      );
      return;
    }

    const image = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      mediaTypes: ["images"],
      quality: 1,
    });

    if (!image.canceled && image.assets[0]) {
      setSelectedImage(toSelectedImage(image.assets[0], "camera"));
      setInputMode("camera");
    }
  }

  async function submitAttempt() {
    if (!token || !selectedCharacter) {
      setMessage("Log in and choose a character first.");
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      let image = selectedImage;
      if (inputMode === "canvas") {
        if (image?.source === "demo_canvas") {
          // Use the prepared demo image instead of capturing the drawing pad.
        } else if (!drawingRef.current?.hasDrawing()) {
          throw new Error(
            "Draw the character on the canvas before submitting.",
          );
        } else {
          const uri = await drawingRef.current.capture();
          image = {
            uri,
            name: `${selectedCharacter.name}_canvas_${Date.now()}.png`,
            type: "image/png",
            source: "canvas",
          };
        }
      }

      if (!image) {
        throw new Error("Choose or draw an image before submitting.");
      }

      setSubmittedImage(image);
      const response = await submitPracticeAttempt(
        apiBaseUrl,
        token,
        selectedCharacter.id,
        selectedMode,
        image,
      );
      setResultDetailsOpen(false);
      setResult(response);
      await refreshAppData();
      setScreen("results");
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Could not submit attempt.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function loadSuggestedRecommendation(): Promise<void> {
    if (!token) {
      return;
    }

    try {
      const response = await fetchPracticeRecommendations(apiBaseUrl, token, 5);
      const pick = response.recommendations[0];
      if (pick) {
        setSelectedCharacterId(pick.character.id);
        setSuggestedReasonText(pick.reason);
        setSuggestedRecommendation(pick);
        return;
      }
    } catch {
      // Keep practice usable offline or if the recommendation endpoint is unavailable.
    }

    const fallbackPick = chooseSuggestedPick(progress);
    if (fallbackPick) {
      setSelectedCharacterId(fallbackPick.item.character.id);
      setSuggestedReasonText(fallbackPick.reason);
      setSuggestedRecommendation(null);
    }
  }

  async function beginPractice(mode: PracticeMode) {
    if (mode === "app_suggested") {
      await loadSuggestedRecommendation();
    } else {
      setSuggestedReasonText(null);
      setSuggestedRecommendation(null);
    }

    setSelectedMode(mode);
    setSelectedImage(null);
    setSubmittedImage(null);
    setResult(null);
    setResultDetailsOpen(false);
    setMessage(null);
    setCharacterSearch("");
    setCharacterPickerOpen(false);
    setScreen("practice");
  }

  async function continuePractice() {
    if (selectedMode === "app_suggested") {
      await loadSuggestedRecommendation();
    }

    setSelectedImage(null);
    setSubmittedImage(null);
    setResult(null);
    setResultDetailsOpen(false);
    setMessage(null);
    drawingRef.current?.clear();
    setScreen("practice");
  }

  async function openCharacterDetail(characterId: number) {
    if (!token) {
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      const detail = await fetchCharacterProgress(
        apiBaseUrl,
        token,
        characterId,
      );
      setCharacterDetail(detail);
      setScreen("character_detail");
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Could not load character profile.",
      );
    } finally {
      setLoading(false);
    }
  }

  function renderAuth() {
    return (
      <LinearGradient
        colors={[THEME.accent, THEME.softAccent, THEME.background]}
        locations={[0, 0.42, 0.62]}
        style={styles.authGradient}
      >
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : undefined}
          style={styles.centerScreen}
        >
          <View style={styles.authContent}>
            <Image
              accessibilityLabel="Ranjana Lipi Handwriting Learner"
              source={APP_LOGO}
              style={styles.brandLogo}
              resizeMode="contain"
            />
            <Text style={[styles.subtitle, styles.authSubtitle]}>
              {authMode === "login"
                ? "Log in to continue learning."
                : "Create your learner account."}
            </Text>

            <TextInput
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="email-address"
              style={[styles.input, styles.authInput]}
              value={email}
              onChangeText={setEmail}
              placeholder="Enter your email"
              placeholderTextColor={THEME.slate}
            />

            <TextInput
              secureTextEntry
              style={[styles.input, styles.authInput]}
              value={password}
              onChangeText={setPassword}
              placeholder="Enter your password"
              placeholderTextColor={THEME.slate}
            />

            <TouchableOpacity
              disabled={loading}
              style={[styles.authPrimaryButton, loading && styles.disabled]}
              onPress={handleAuth}
            >
              <Text style={styles.authPrimaryButtonText}>
                {loading
                  ? "Please wait..."
                  : authMode === "login"
                    ? "Log In"
                    : "Create your account"}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              disabled={loading}
              style={[styles.googleButton, loading && styles.disabled]}
              onPress={() => void handleGoogleAuth()}
            >
              <Text style={styles.googleButtonText}>Continue with Google</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.linkButton}
              onPress={() => {
                setAuthMode(authMode === "login" ? "register" : "login");
                setMessage(null);
              }}
            >
              <Text style={styles.linkText}>
                {authMode === "login"
                  ? "Create a new account"
                  : "Already have an account? Log in"}
              </Text>
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </LinearGradient>
    );
  }

  function renderHome() {
    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Header
          title="Ranjana Lipi Handwriting Learner"
        />

        <Text style={styles.homeEyebrow}>Learn at your pace</Text>
        <View style={styles.modeGrid}>
          {PRACTICE_MODES.map((mode) => (
            <Pressable
              key={mode.value}
              style={styles.modeButton}
              onPress={() => void beginPractice(mode.value)}
            >
              <Text style={styles.modeTitle}>{mode.label}</Text>
              <Text style={styles.modeText}>
                {mode.value === "free_practice"
                  ? "Practice character of your liking."
                  : "Learn characters with spaced repetition."}
              </Text>
            </Pressable>
          ))}
        </View>

        <View style={styles.rowBetween}>
          <Text style={styles.homeSectionTitle}>Progress</Text>
          <TouchableOpacity onPress={() => setScreen("progress")}>
            <Text style={styles.linkText}>Open insights</Text>
          </TouchableOpacity>
        </View>
        {progress.slice(0, 5).map((item) => (
          <ProgressRow
            key={item.character.id}
            item={item}
            onPress={() => void openCharacterDetail(item.character.id)}
          />
        ))}
      </ScrollView>
    );
  }

  function renderPractice() {
    const practiceTitle =
      selectedMode === "free_practice"
        ? "Free Practice"
        : selectedMode === "app_suggested"
          ? "Suggestive Learning"
          : "Practice Attempt";

    return (
      <View style={styles.practiceScreen}>
        <ScrollView
          contentContainerStyle={styles.practiceScrollContent}
          keyboardShouldPersistTaps="handled"
        >
          <TopNav title={practiceTitle} onBack={() => setScreen("home")} />

          {selectedCharacter ? (
            <View style={styles.practiceGlyphBox}>
              <Image
                source={{
                  uri: characterGlyphUri(apiBaseUrl, selectedCharacter.name),
                }}
                style={styles.practiceGlyphImage}
                resizeMode="contain"
              />
              <Text style={styles.practiceDevanagariLabel}>
                {characterDisplayLabel(selectedCharacter)}
              </Text>
              {selectedMode === "free_practice" ? (
                <TouchableOpacity
                  style={styles.changeCharacterButton}
                  onPress={() => setCharacterPickerOpen((current) => !current)}
                >
                  <Text style={styles.changeCharacterText}>
                    {characterPickerOpen ? "Done" : "Change Character"}
                  </Text>
                </TouchableOpacity>
              ) : null}
            </View>
          ) : null}

          {selectedMode === "app_suggested" && suggestedRecommendation ? (
            <View style={styles.recommendationNote}>
              <Text style={styles.recommendationNoteText}>
                {suggestedReasonText ??
                  "Recommended for today's spaced repetition practice."}
              </Text>
            </View>
          ) : null}

          {selectedMode === "free_practice" && characterPickerOpen ? (
            <View style={styles.characterPickerPanel}>
              <View style={styles.characterSearchBar}>
                <Text style={styles.characterSearchIcon}>⌕</Text>
                <TextInput
                  autoCapitalize="none"
                  autoCorrect={false}
                  placeholder="Search by name or letter"
                  placeholderTextColor={THEME.slate}
                  style={styles.characterSearchInput}
                  value={characterSearch}
                  onChangeText={setCharacterSearch}
                />
              </View>
              <ScrollView
                nestedScrollEnabled
                style={styles.characterPickerList}
                contentContainerStyle={styles.characterPickerListContent}
                keyboardShouldPersistTaps="handled"
              >
                {filteredCharacterSections.map((section) => (
                  <View key={section.key} style={styles.characterSection}>
                    <Text style={styles.characterSectionTitle}>
                      {section.title}
                    </Text>
                    <View style={styles.characterGrid}>
                      {section.characters.map((character) => (
                        <TouchableOpacity
                          key={character.id}
                          style={[
                            styles.characterChip,
                            selectedCharacter?.id === character.id &&
                              styles.selectedChip,
                          ]}
                          onPress={() => {
                            setSelectedCharacterId(character.id);
                            setCharacterPickerOpen(false);
                            setCharacterSearch("");
                          }}
                        >
                          <Text
                            style={[
                              styles.characterName,
                              selectedCharacter?.id === character.id &&
                                styles.selectedChipText,
                            ]}
                          >
                            {characterDisplayLabel(character)}
                          </Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  </View>
                ))}
              </ScrollView>
              {filteredCharacters.length === 0 ? (
                <Text style={styles.emptyText}>
                  No characters match that search.
                </Text>
              ) : null}
            </View>
          ) : null}

          <Text style={styles.sectionTitle}>How would you like to input?</Text>
          <View style={styles.inputActions}>
            <SecondaryButton
              label="Gallery"
              icon="▧"
              onPress={pickFromGallery}
              active={inputMode === "gallery"}
            />
            <SecondaryButton
              label="Camera"
              icon="◉"
              onPress={takePhoto}
              active={inputMode === "camera"}
            />
            <SecondaryButton
              label="Canvas"
              icon="✎"
              onPress={() => {
                setInputMode("canvas");
                setSelectedImage(null);
              }}
              active={inputMode === "canvas"}
            />
          </View>

          {inputMode === "canvas" ? (
            <View style={styles.canvasWrap}>
              {selectedCharacter?.name === "a" ? (
                <TouchableOpacity
                  style={styles.demoCanvasButton}
                  onPress={() => {
                    setSelectedImage({
                      uri: demoCanvasUri(apiBaseUrl, "a"),
                      name: "a_canvas_demo_high_score.png",
                      type: "image/png",
                      source: "demo_canvas",
                    });
                  }}
                >
                  <Text style={styles.demoCanvasButtonText}>
                    Use Demo Image
                  </Text>
                </TouchableOpacity>
              ) : null}
              {selectedImage?.source === "demo_canvas" ? (
                <View style={styles.previewWrap}>
                  <Image
                    source={{ uri: selectedImage.uri }}
                    style={styles.previewImage}
                    resizeMode="contain"
                  />
                  <Text style={styles.previewText}>
                    Demo canvas image selected.
                  </Text>
                </View>
              ) : null}
              <DrawingCanvas ref={drawingRef} />
            </View>
          ) : selectedImage ? (
            <View style={styles.previewWrap}>
              <Image
                source={{ uri: selectedImage.uri }}
                style={styles.previewImage}
                resizeMode="contain"
              />
              <Text style={styles.previewText}>{selectedImage.name}</Text>
            </View>
          ) : (
            <EmptyState
              title="No input selected"
              text="Choose a gallery photo, take a camera photo, or draw on the canvas."
            />
          )}
        </ScrollView>

        <View style={styles.stickySubmitBar}>
          <PrimaryButton
            disabled={loading}
            label={loading ? "Analyzing..." : "Submit Attempt"}
            onPress={submitAttempt}
          />
        </View>
      </View>
    );
  }

  function renderResults() {
    const feedback = result?.region_feedback ?? null;
    const score = result?.overall_score ?? feedback?.overall_score ?? null;
    const isWrongCharacter = Boolean(feedback?.wrong_character);
    const isBlockedFeedback = Boolean(
      feedback?.wrong_character ||
        feedback?.invalid_input ||
        feedback?.insufficient_input,
    );
    const canShowRegionFeedback = !(
      feedback?.wrong_character ||
      feedback?.invalid_input ||
      feedback?.insufficient_input
    );
    const summary = regionSummary(feedback);
    const predictedLabel =
      feedback?.predicted_class && DEVANAGARI_LABELS[feedback.predicted_class]
        ? DEVANAGARI_LABELS[feedback.predicted_class]
        : null;
    const referenceUri = selectedCharacter
      ? characterGlyphUri(apiBaseUrl, selectedCharacter.name)
      : null;

    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Feedback" onBack={() => setScreen("practice")} />
        <View style={styles.scorePanel}>
          <Text style={styles.scoreLabel}>{resultStatusText(score, feedback)}</Text>
          <Text style={[styles.scoreValue, { color: scoreColor(score) }]}>
            {typeof score === "number" ? `${score.toFixed(1)}%` : "--"}
          </Text>
          <Text
            style={[
              styles.resultFeedbackMessage,
              isBlockedFeedback && styles.resultFeedbackWarning,
            ]}
          >
            {learnerFeedbackText(feedback)}
          </Text>
        </View>

        {isWrongCharacter && selectedCharacter ? (
          <View style={styles.matchStatusCard}>
            <Text style={styles.matchStatusLabel}>Character match</Text>
            <Text style={styles.matchStatusText}>
              Expected {characterDisplayLabel(selectedCharacter)}
              {predictedLabel ? `, detected ${predictedLabel}` : ""}
            </Text>
          </View>
        ) : null}

        <Text style={styles.sectionTitle}>Comparison</Text>
        <View style={styles.comparisonRow}>
          <View style={styles.comparisonPanel}>
            <Text style={styles.comparisonLabel}>Reference</Text>
            {referenceUri ? (
              <Image
                source={{ uri: referenceUri }}
                style={styles.comparisonImage}
                resizeMode="contain"
              />
            ) : (
              <View style={styles.comparisonPlaceholder}>
                <Text style={styles.emptyText}>No reference</Text>
              </View>
            )}
          </View>
          <View style={styles.comparisonPanel}>
            <Text style={styles.comparisonLabel}>Your writing</Text>
            {submittedImage ? (
              <Image
                source={{ uri: submittedImage.uri }}
                style={styles.comparisonImage}
                resizeMode="contain"
              />
            ) : (
              <View style={styles.comparisonPlaceholder}>
                <Text style={styles.emptyText}>No input</Text>
              </View>
            )}
          </View>
        </View>

        {canShowRegionFeedback ? (
          <>
            <Text style={styles.sectionTitle}>Region guide</Text>
            <RegionGrid feedback={feedback} />

            <View style={styles.feedbackSummaryCard}>
              <View style={styles.feedbackSummaryItem}>
                <Text style={styles.feedbackSummaryLabel}>Main focus</Text>
                <Text style={styles.feedbackSummaryValue}>{summary.main}</Text>
              </View>
              <View style={styles.feedbackSummaryDivider} />
              <View style={styles.feedbackSummaryItem}>
                <Text style={styles.feedbackSummaryLabel}>Also check</Text>
                <Text style={styles.feedbackSummaryValue}>
                  {summary.secondary}
                </Text>
              </View>
              <View style={styles.feedbackSummaryDivider} />
              <View style={styles.feedbackSummaryItem}>
                <Text style={styles.feedbackSummaryLabel}>Looks good</Text>
                <Text style={styles.feedbackSummaryValue}>{summary.clear}</Text>
              </View>
            </View>
          </>
        ) : null}

        <View style={styles.nextStepCard}>
          <Text style={styles.nextStepTitle}>Next step</Text>
          <Text style={styles.nextStepText}>{coachingSuggestion(feedback)}</Text>
        </View>

        <TouchableOpacity
          activeOpacity={0.85}
          style={styles.whyResultToggle}
          onPress={() => setResultDetailsOpen((open) => !open)}
        >
          <Text style={styles.whyResultToggleText}>Why this result?</Text>
          <Text style={styles.whyResultToggleIcon}>
            {resultDetailsOpen ? "-" : "+"}
          </Text>
        </TouchableOpacity>
        {resultDetailsOpen ? (
          <View style={styles.whyResultPanel}>
            <Text style={styles.whyResultText}>
              {technicalResultText(feedback)}
            </Text>
          </View>
        ) : null}

        <View style={styles.resultActions}>
          <SecondaryButton
            label={
              selectedMode === "app_suggested" ? "Next Suggested" : "Try Again"
            }
            onPress={() => void continuePractice()}
          />
          <SecondaryButton
            label="Progress"
            onPress={() => setScreen("progress")}
          />
        </View>
      </ScrollView>
    );
  }

  function renderProgress() {
    const stats = profile?.stats;

    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Insights" onBack={() => setScreen("home")} />
        <View style={styles.insightStatsGrid}>
          <View style={styles.insightStatBox}>
            <Text style={styles.insightStatValue}>
              {stats?.total_attempts ?? 0}
            </Text>
            <Text style={styles.insightStatLabel}>Attempts</Text>
          </View>
          <View style={styles.insightStatBox}>
            <Text style={styles.insightStatValue}>
              {stats?.practiced_characters ?? 0}
            </Text>
            <Text style={styles.insightStatLabel}>Practiced</Text>
          </View>
          <View style={styles.insightStatBox}>
            <Text style={styles.insightStatValue}>
              {stats?.mastered_characters ?? 0}
            </Text>
            <Text style={styles.insightStatLabel}>Mastered</Text>
          </View>
          <View style={styles.insightStatBox}>
            <Text style={styles.insightStatValue}>
              {stats?.current_streak_days ?? 0}
            </Text>
            <Text style={styles.insightStatLabel}>Day Streak</Text>
          </View>
        </View>

        <View style={styles.insightScoreRow}>
          <View>
            <Text style={styles.insightScoreLabel}>Average Score</Text>
            <Text
              style={[
                styles.insightScoreValue,
                { color: scoreColor(stats?.average_score) },
              ]}
            >
              {scoreText(stats?.average_score)}
            </Text>
          </View>
          <View>
            <Text style={styles.insightScoreLabel}>Best Score</Text>
            <Text
              style={[
                styles.insightScoreValue,
                { color: scoreColor(stats?.best_score) },
              ]}
            >
              {scoreText(stats?.best_score)}
            </Text>
          </View>
          <View>
            <Text style={styles.insightScoreLabel}>Longest Streak</Text>
            <Text style={styles.insightScoreValue}>
              {stats?.longest_streak_days ?? 0}d
            </Text>
          </View>
        </View>

        <Text style={styles.profileSectionTitle}>Character Progress</Text>
        {progressSections.map((section) => (
          <View key={section.key} style={styles.progressSection}>
            <Text style={styles.progressSectionTitle}>{section.title}</Text>
            {section.items.map((item) => (
              <ProgressRow
                key={item.character.id}
                item={item}
                large
                onPress={() => void openCharacterDetail(item.character.id)}
              />
            ))}
          </View>
        ))}
      </ScrollView>
    );
  }

  function renderAttemptCard(attempt: Attempt) {
    return (
      <View key={attempt.id} style={styles.attemptCard}>
        <Image
          source={{ uri: attemptImageUri(apiBaseUrl, attempt) }}
          style={styles.attemptThumb}
          resizeMode="contain"
        />
        <View style={styles.attemptBody}>
          <Text style={styles.attemptTitle}>
            {attemptCharacterLabel(characters, attempt)}
          </Text>
          <Text style={styles.attemptMeta}>
            {formatAttemptDate(attempt.created_at)} |{" "}
            {attempt.mode.replace("_", " ")}
          </Text>
          <Text
            style={[
              styles.attemptScore,
              { color: scoreColor(attempt.overall_score) },
            ]}
          >
            {scoreText(attempt.overall_score)}
          </Text>
          <Text style={styles.attemptRegion}>
            {topRegionSummary(attempt.region_feedback)}
          </Text>
        </View>
      </View>
    );
  }

  function renderHistory() {
    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Attempt History" onBack={() => setScreen("profile")} />
        <View style={styles.historyIntroCard}>
          <Text style={styles.historyIntroTitle}>Recent practice</Text>
          <Text style={styles.historyIntroText}>
            Review submitted work, scores, and the strongest feedback region.
          </Text>
        </View>
        {attemptHistory.length > 0 ? (
          attemptHistory.map((attempt) => renderAttemptCard(attempt))
        ) : (
          <EmptyState
            title="No attempts yet"
            text="Practice a character once and your history will appear here."
          />
        )}
      </ScrollView>
    );
  }

  function renderCharacterDetail() {
    const detail = characterDetail;
    if (!detail) {
      return (
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <TopNav
            title="Character Profile"
            onBack={() => setScreen("progress")}
          />
          <EmptyState
            title="No character loaded"
            text="Go back to Insights and choose a character again."
          />
        </ScrollView>
      );
    }

    const progressItem: ProgressDashboardItem = {
      character: detail.character,
      progress: detail.progress,
    };
    const intervalHours = ankiStyleIntervalHours(progressItem);
    const elapsedHours = hoursSince(
      detail.progress?.last_practiced_at,
      Date.now(),
    );
    const reviewStatus = suggestedReason(
      progressItem,
      intervalHours,
      elapsedHours,
    );

    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav
          title="Character Profile"
          onBack={() => setScreen("progress")}
        />
        <View style={styles.characterDetailHero}>
          <Image
            source={{
              uri: characterGlyphUri(apiBaseUrl, detail.character.name),
            }}
            style={styles.characterDetailGlyph}
            resizeMode="contain"
          />
          <Text style={styles.characterDetailDevanagari}>
            {characterDisplayLabel(detail.character)}
          </Text>
        </View>

        <View style={styles.profileStatsGrid}>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>
              {detail.progress?.attempts_count ?? 0}
            </Text>
            <Text style={styles.profileStatLabel}>Attempts</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>
              {scoreText(detail.progress?.best_score)}
            </Text>
            <Text style={styles.profileStatLabel}>Best Score</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>
              {detail.progress?.mastered ? "Yes" : "No"}
            </Text>
            <Text style={styles.profileStatLabel}>Mastered</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>{reviewStatus}</Text>
            <Text style={styles.profileStatLabel}>Review Status</Text>
          </View>
        </View>

        <PrimaryButton
          label="Practice This Character"
          onPress={() => {
            setSelectedCharacterId(detail.character.id);
            setSelectedMode("free_practice");
            setCharacterPickerOpen(false);
            setSelectedImage(null);
            setSubmittedImage(null);
            setResult(null);
            setScreen("practice");
          }}
        />

        <Text style={styles.sectionTitle}>Recent Attempts</Text>
        {detail.attempts.length > 0 ? (
          detail.attempts
            .slice(0, 10)
            .map((attempt) => renderAttemptCard(attempt))
        ) : (
          <EmptyState
            title="No attempts yet"
            text="Practice this character once to start building progress."
          />
        )}
      </ScrollView>
    );
  }

  function renderProfile() {
    const displayName =
      profile?.user.display_name ??
      profile?.user.email ??
      user?.display_name ??
      user?.email ??
      "Student";
    const emailAddress = profile?.user.email ?? user?.email ?? "";
    const heatmap = profile?.heatmap ?? [];

    return (
      <ScrollView
        contentContainerStyle={[
          styles.profileScreenContent,
          { paddingTop: Math.max(Constants.statusBarHeight + 18, 42) },
        ]}
      >
        <View style={styles.profileTopBar}>
          <TouchableOpacity
            accessibilityLabel="Go back"
            style={styles.profileIconButton}
            onPress={() => setScreen("home")}
          >
            <View style={styles.profileBackIcon}>
              <View
                style={[styles.profileBackIconLine, styles.profileBackIconLineTop]}
              />
              <View
                style={[
                  styles.profileBackIconLine,
                  styles.profileBackIconLineBottom,
                ]}
              />
            </View>
          </TouchableOpacity>
          <Text style={styles.profileScreenTitle}>Profile</Text>
          <View style={styles.profileIconButton} />
        </View>

        <View style={styles.profileHeader}>
          <View style={styles.profileAvatar}>
            <Text style={styles.profileAvatarText}>
              {displayName.slice(0, 1).toUpperCase()}
            </Text>
          </View>
          <View style={styles.profileIdentity}>
            <Text style={styles.profileName}>{displayName}</Text>
            <Text style={styles.profileEmail}>{emailAddress}</Text>
          </View>
        </View>

        <Text style={styles.profileSectionTitle}>Your logs</Text>
        <View style={styles.heatmapPanel}>
          <View style={styles.heatmapGrid}>
            {heatmap.map((day) => (
              <View
                key={day.date}
                style={[
                  styles.heatmapCell,
                  { backgroundColor: heatmapColor(day.attempts_count) },
                ]}
              />
            ))}
          </View>
          <View style={styles.heatmapLegend}>
            <Text style={styles.heatmapLegendText}>Less</Text>
            {[0, 1, 2, 4, 6].map((count) => (
              <View
                key={count}
                style={[
                  styles.heatmapLegendCell,
                  { backgroundColor: heatmapColor(count) },
                ]}
              />
            ))}
            <Text style={styles.heatmapLegendText}>More</Text>
          </View>
        </View>

        <Text style={styles.profileSectionTitle}>Account</Text>
        <View style={styles.accountPanel}>
          <Text style={styles.accountLabel}>Display name</Text>
          <TextInput
            autoCapitalize="words"
            autoCorrect={false}
            placeholder="Your name"
            placeholderTextColor={THEME.slate}
            style={styles.accountInput}
            value={accountDisplayName}
            onChangeText={setAccountDisplayName}
          />

          <Text style={styles.accountLabel}>Email</Text>
          <TextInput
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="email-address"
            placeholder="name@example.com"
            placeholderTextColor={THEME.slate}
            style={styles.accountInput}
            value={accountEmail}
            onChangeText={setAccountEmail}
          />

          <TouchableOpacity
            disabled={loading}
            style={[styles.accountPrimaryButton, loading && styles.disabled]}
            onPress={handleAccountUpdate}
          >
            <Text style={styles.accountPrimaryButtonText}>Save Changes</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.accountPanel}>
          <Text style={styles.accountLabel}>Current password</Text>
          <TextInput
            secureTextEntry
            placeholder="Current password"
            placeholderTextColor={THEME.slate}
            style={styles.accountInput}
            value={currentPassword}
            onChangeText={setCurrentPassword}
          />

          <Text style={styles.accountLabel}>New password</Text>
          <TextInput
            secureTextEntry
            placeholder="New password"
            placeholderTextColor={THEME.slate}
            style={styles.accountInput}
            value={newPassword}
            onChangeText={setNewPassword}
          />

          <TouchableOpacity
            disabled={loading}
            style={[styles.accountSecondaryButton, loading && styles.disabled]}
            onPress={handlePasswordChange}
          >
            <Text style={styles.accountSecondaryButtonText}>
              Change Password
            </Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          disabled={loading}
          style={[styles.profileDeactivateButton, loading && styles.disabled]}
          onPress={confirmDeactivateAccount}
        >
          <Text style={styles.profileDeactivateButtonText}>
            Deactivate Account
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.profileLogoutButton} onPress={handleLogout}>
          <Text style={styles.profileLogoutButtonText}>Logout</Text>
        </TouchableOpacity>

      </ScrollView>
    );
  }

  return (
    <View style={styles.app}>
      <StatusBar style="dark" />
      {screen === "auth"
        ? renderAuth()
        : screen === "home"
          ? renderHome()
          : screen === "practice"
            ? renderPractice()
            : screen === "results"
              ? renderResults()
              : screen === "progress"
                ? renderProgress()
                : screen === "profile"
                  ? renderProfile()
                  : screen === "history"
                    ? renderHistory()
                    : renderCharacterDetail()}
      {screen !== "auth" ? (
        <BottomNav
          activeScreen={screen}
          onHome={() => setScreen("home")}
          onPractice={() => {
            setSelectedMode("free_practice");
            setScreen("practice");
          }}
          onInsights={() => setScreen("progress")}
          onProfile={() => setScreen("profile")}
        />
      ) : null}
      {loading && screen !== "auth" ? (
        <View style={styles.loadingOverlay}>
          <View style={styles.loadingCard}>
            <ActivityIndicator color={THEME.accent} />
            <Text style={styles.loadingTitle}>Working on it</Text>
            <Text style={styles.loadingText}>
              Please wait while the app updates your practice data.
            </Text>
          </View>
        </View>
      ) : null}
      {message ? (
        <TouchableOpacity
          style={styles.message}
          onPress={() => setMessage(null)}
        >
          <Text style={styles.messageText}>{message}</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}

function Header({ title }: { title: string }) {
  return (
    <View style={styles.header}>
      <View style={styles.headerBrand}>
        <Image
          accessibilityLabel={title}
          source={APP_LOGO}
          style={styles.headerLogo}
          resizeMode="contain"
        />
      </View>
    </View>
  );
}

function TopNav({ title, onBack }: { title: string; onBack: () => void }) {
  return (
    <View style={[styles.header, styles.topNavHeader]}>
      <TouchableOpacity
        accessibilityLabel="Go back"
        style={styles.backIconButton}
        onPress={onBack}
      >
        <View style={styles.profileBackIcon}>
          <View
            style={[styles.profileBackIconLine, styles.profileBackIconLineTop]}
          />
          <View
            style={[
              styles.profileBackIconLine,
              styles.profileBackIconLineBottom,
            ]}
          />
        </View>
      </TouchableOpacity>
      <Text style={styles.screenTitle}>{title}</Text>
      <View style={styles.navSpacer} />
    </View>
  );
}

function BottomNav({
  activeScreen,
  onHome,
  onPractice,
  onInsights,
  onProfile,
}: {
  activeScreen: Screen;
  onHome: () => void;
  onPractice: () => void;
  onInsights: () => void;
  onProfile: () => void;
}) {
  const items = [
    {
      key: "home",
      label: "Home",
      icon: "⌂",
      active: activeScreen === "home",
      onPress: onHome,
    },
    {
      key: "practice",
      label: "Practice",
      icon: "✎",
      active: activeScreen === "practice" || activeScreen === "results",
      onPress: onPractice,
    },
    {
      key: "insights",
      label: "Insights",
      icon: "▥",
      active:
        activeScreen === "progress" ||
        activeScreen === "character_detail",
      onPress: onInsights,
    },
    {
      key: "profile",
      label: "Profile",
      icon: "●",
      active: activeScreen === "profile",
      onPress: onProfile,
    },
  ];

  return (
    <View style={styles.bottomNavShell}>
      <View style={styles.bottomNav}>
        {items.map((item) => (
          <TouchableOpacity
            key={item.key}
            accessibilityRole="button"
            accessibilityState={{ selected: item.active }}
            style={[
              styles.bottomNavItem,
              item.active && styles.bottomNavItemActive,
            ]}
            onPress={item.onPress}
          >
            <Text
              style={[
                styles.bottomNavIcon,
                item.active && styles.bottomNavIconActive,
              ]}
            >
              {item.icon}
            </Text>
            <Text
              style={[
                styles.bottomNavLabel,
                item.active && styles.bottomNavLabelActive,
              ]}
            >
              {item.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

function PrimaryButton({
  label,
  onPress,
  disabled,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
}) {
  return (
    <TouchableOpacity
      disabled={disabled}
      style={[styles.primaryButton, disabled && styles.disabled]}
      onPress={onPress}
    >
      <Text style={styles.primaryButtonText}>{label}</Text>
    </TouchableOpacity>
  );
}

function SecondaryButton({
  label,
  icon,
  onPress,
  active,
}: {
  label: string;
  icon?: string;
  onPress: () => void;
  active?: boolean;
}) {
  return (
    <TouchableOpacity
      style={[styles.secondaryButton, active && styles.secondaryButtonActive]}
      onPress={onPress}
    >
      {icon ? (
        <Text
          style={[
            styles.secondaryButtonIcon,
            active && styles.secondaryButtonTextActive,
          ]}
        >
          {icon}
        </Text>
      ) : null}
      <Text
        style={[
          styles.secondaryButtonText,
          active && styles.secondaryButtonTextActive,
        ]}
      >
        {label}
      </Text>
    </TouchableOpacity>
  );
}

function EmptyState({ title, text }: { title: string; text: string }) {
  return (
    <View style={styles.emptyStateCard}>
      <Text style={styles.emptyStateTitle}>{title}</Text>
      <Text style={styles.emptyStateText}>{text}</Text>
    </View>
  );
}

function ProgressRow({
  item,
  large,
  onPress,
}: {
  item: ProgressDashboardItem;
  large?: boolean;
  onPress?: () => void;
}) {
  const bestScore = item.progress?.best_score ?? null;
  const Container = onPress ? TouchableOpacity : View;
  return (
    <Container
      style={[styles.progressRow, large && styles.progressRowLarge]}
      onPress={onPress}
    >
      <View>
        <Text style={styles.progressName}>
          {characterDisplayLabel(item.character)}
        </Text>
        <Text style={styles.progressMeta}>
          {item.progress?.attempts_count ?? 0} attempts
          {item.progress?.mastered ? " | mastered" : ""}
        </Text>
      </View>
      {typeof bestScore === "number" ? (
        <Text style={[styles.progressScore, { color: scoreColor(bestScore) }]}>
          {bestScore.toFixed(1)}%
        </Text>
      ) : (
        <View style={styles.progressNewBadge}>
          <Text style={styles.progressNewBadgeText}>New</Text>
        </View>
      )}
    </Container>
  );
}

const styles = StyleSheet.create({
  app: {
    flex: 1,
    backgroundColor: THEME.background,
  },
  centerScreen: {
    flex: 1,
    justifyContent: "center",
    padding: 20,
  },
  authGradient: {
    flex: 1,
  },
  authContent: {
    width: "100%",
  },
  brand: {
    color: THEME.ink,
    fontSize: 30,
    fontWeight: "800",
    marginBottom: 6,
  },
  brandLogo: {
    alignSelf: "center",
    height: 74,
    marginBottom: 8,
    transform: [{ translateY: -44 }],
    width: 274,
  },
  subtitle: {
    color: THEME.slate,
    fontSize: 14,
  },
  authSubtitle: {
    marginBottom: 28,
    textAlign: "center",
    transform: [{ translateY: -44 }],
  },
  label: {
    color: THEME.ink,
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 6,
    marginTop: 14,
  },
  input: {
    backgroundColor: THEME.surface,
    borderColor: THEME.ink,
    borderRadius: 8,
    borderWidth: 1.5,
    color: THEME.ink,
    minHeight: 46,
    paddingHorizontal: 12,
  },
  authInput: {
    borderColor: "#D9DEE2",
    borderRadius: 26,
    borderWidth: 1,
    color: THEME.ink,
    fontSize: 15,
    marginTop: 14,
    minHeight: 54,
    paddingHorizontal: 22,
  },
  fieldHint: {
    color: THEME.slate,
    fontSize: 12,
    marginTop: 6,
  },
  scrollContent: {
    padding: 22,
    paddingBottom: 124,
    paddingTop: Math.max(Constants.statusBarHeight + 24, 44),
  },
  profileScreenContent: {
    padding: 24,
    paddingBottom: 132,
  },
  practiceScreen: {
    flex: 1,
  },
  practiceScrollContent: {
    padding: 22,
    paddingBottom: 190,
    paddingTop: Math.max(Constants.statusBarHeight + 24, 48),
  },
  header: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
  },
  topNavHeader: {
    alignItems: "center",
    marginBottom: 22,
  },
  headerBrand: {
    alignItems: "center",
    flex: 1,
  },
  screenTitle: {
    color: THEME.ink,
    fontSize: 30,
    fontWeight: "900",
    lineHeight: 48,
  },
  headerLogo: {
    height: 36,
    marginLeft: 0,
    width: 96,
  },
  smallButton: {
    borderColor: THEME.muted,
    borderRadius: 7,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  smallButtonText: {
    color: THEME.ink,
    fontWeight: "700",
  },
  backIconButton: {
    alignItems: "center",
    height: 48,
    justifyContent: "center",
    width: 48,
  },
  navSpacer: {
    width: 42,
  },
  sectionTitle: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "800",
    marginBottom: 12,
    marginTop: 12,
  },
  homeEyebrow: {
    color: THEME.ink,
    fontSize: 20,
    fontWeight: "900",
    marginTop: 22,
    marginBottom: 16,
  },
  homeSectionTitle: {
    color: THEME.ink,
    fontSize: 22,
    fontWeight: "900",
  },
  modeGrid: {
    gap: 16,
  },
  modeButton: {
    backgroundColor: THEME.surface,
    borderRadius: 26,
    minHeight: 92,
    paddingHorizontal: 22,
    paddingVertical: 22,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
    elevation: 3,
  },
  modeTitle: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "900",
  },
  modeText: {
    color: THEME.slate,
    fontSize: 14,
    lineHeight: 20,
    marginTop: 6,
  },
  characterGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    paddingBottom: 4,
  },
  characterChip: {
    backgroundColor: THEME.surface,
    borderRadius: 22,
    elevation: 1,
    flexBasis: "29%",
    flexGrow: 1,
    minHeight: 86,
    paddingHorizontal: 12,
    paddingVertical: 14,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.06,
    shadowRadius: 12,
  },
  practiceGlyphBox: {
    alignItems: "center",
    alignSelf: "center",
    backgroundColor: THEME.surface,
    borderRadius: 30,
    justifyContent: "center",
    marginBottom: 18,
    minHeight: 286,
    paddingHorizontal: 22,
    paddingVertical: 22,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
    elevation: 3,
    width: "100%",
  },
  practiceGlyphImage: {
    height: 160,
    width: "92%",
  },
  practiceDevanagariLabel: {
    color: THEME.ink,
    fontSize: 42,
    fontWeight: "900",
    marginTop: 12,
  },
  selectedCharacterPanel: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    padding: 14,
  },
  selectedCharacterLabel: {
    color: THEME.slate,
    fontSize: 12,
    fontWeight: "800",
  },
  selectedCharacterName: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "900",
    marginTop: 2,
  },
  suggestedReasonText: {
    color: THEME.slate,
    fontSize: 12,
    fontWeight: "700",
    marginTop: 3,
  },
  recommendationPanel: {
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    marginTop: 10,
    padding: 14,
  },
  recommendationTitle: {
    color: THEME.ink,
    fontSize: 15,
    fontWeight: "900",
  },
  recommendationText: {
    color: THEME.slate,
    lineHeight: 20,
    marginTop: 6,
  },
  recommendationStats: {
    flexDirection: "row",
    gap: 8,
    marginTop: 12,
  },
  recommendationStat: {
    backgroundColor: THEME.softMuted,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    padding: 10,
  },
  recommendationStatValue: {
    color: THEME.ink,
    fontSize: 14,
    fontWeight: "900",
  },
  recommendationStatLabel: {
    color: THEME.slate,
    fontSize: 11,
    fontWeight: "800",
    marginTop: 4,
  },
  recommendationNote: {
    backgroundColor: THEME.softAccent,
    borderRadius: 20,
    marginBottom: 22,
    paddingHorizontal: 18,
    paddingVertical: 14,
  },
  recommendationNoteText: {
    color: THEME.ink,
    fontSize: 14,
    fontWeight: "800",
    lineHeight: 20,
  },
  changeCharacterButton: {
    backgroundColor: THEME.accent,
    borderRadius: 22,
    marginTop: 14,
    paddingHorizontal: 18,
    paddingVertical: 10,
  },
  changeCharacterText: {
    color: THEME.surface,
    fontSize: 14,
    fontWeight: "800",
  },
  characterPickerPanel: {
    backgroundColor: THEME.surface,
    borderRadius: 26,
    elevation: 2,
    marginBottom: 22,
    padding: 16,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.07,
    shadowRadius: 16,
  },
  characterPickerList: {
    maxHeight: 372,
    marginTop: 16,
  },
  characterPickerListContent: {
    paddingBottom: 4,
  },
  characterSection: {
    marginBottom: 18,
  },
  characterSectionTitle: {
    color: THEME.ink,
    fontSize: 14,
    fontWeight: "900",
    marginBottom: 10,
    textTransform: "uppercase",
  },
  characterSearchBar: {
    alignItems: "center",
    backgroundColor: THEME.softMuted,
    borderRadius: 28,
    flexDirection: "row",
    minHeight: 54,
    paddingHorizontal: 18,
  },
  characterSearchIcon: {
    color: THEME.accent,
    fontSize: 24,
    fontWeight: "900",
    lineHeight: 26,
    marginRight: 10,
  },
  characterSearchInput: {
    color: THEME.ink,
    flex: 1,
    fontSize: 16,
    fontWeight: "700",
    minHeight: 54,
    padding: 0,
  },
  selectedChip: {
    backgroundColor: THEME.accent,
  },
  characterName: {
    color: THEME.ink,
    fontSize: 26,
    fontWeight: "900",
  },
  selectedChipText: {
    color: THEME.surface,
  },
  progressSection: {
    marginBottom: 18,
  },
  progressSectionTitle: {
    color: THEME.slate,
    fontSize: 13,
    fontWeight: "900",
    marginBottom: 10,
    textTransform: "uppercase",
  },
  rowBetween: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 14,
    marginTop: 32,
  },
  linkButton: {
    alignItems: "center",
    marginTop: 14,
  },
  linkText: {
    color: THEME.accent,
    fontSize: 14,
    fontWeight: "800",
  },
  segmented: {
    backgroundColor: THEME.softAccent,
    borderRadius: 8,
    flexDirection: "row",
    padding: 4,
  },
  segmentButton: {
    alignItems: "center",
    borderRadius: 6,
    flex: 1,
    paddingVertical: 10,
  },
  segmentButtonActive: {
    backgroundColor: THEME.surface,
  },
  segmentText: {
    color: THEME.slate,
    fontSize: 12,
    fontWeight: "800",
  },
  segmentTextActive: {
    color: THEME.ink,
  },
  inputActions: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 20,
  },
  canvasWrap: {
    alignItems: "center",
    marginBottom: 18,
    width: "100%",
  },
  demoCanvasButton: {
    alignItems: "center",
    backgroundColor: THEME.accent,
    borderRadius: 26,
    elevation: 2,
    marginBottom: 14,
    minHeight: 54,
    justifyContent: "center",
    paddingVertical: 12,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
    width: "100%",
  },
  demoCanvasButtonText: {
    color: THEME.surface,
    fontWeight: "900",
  },
  previewWrap: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 28,
    elevation: 2,
    marginBottom: 18,
    padding: 16,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
    width: "100%",
  },
  previewImage: {
    height: 240,
    width: "100%",
  },
  previewText: {
    color: THEME.slate,
    fontSize: 12,
    marginTop: 8,
  },
  emptyText: {
    color: THEME.slate,
    marginBottom: 18,
  },
  emptyStateCard: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 28,
    elevation: 2,
    marginBottom: 18,
    paddingHorizontal: 22,
    paddingVertical: 24,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.06,
    shadowRadius: 14,
  },
  emptyStateTitle: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "900",
    textAlign: "center",
  },
  emptyStateText: {
    color: THEME.slate,
    fontSize: 14,
    fontWeight: "700",
    lineHeight: 21,
    marginTop: 8,
    textAlign: "center",
  },
  primaryButton: {
    alignItems: "center",
    backgroundColor: THEME.accent,
    borderRadius: 28,
    marginTop: 16,
    minHeight: 56,
    justifyContent: "center",
    paddingVertical: 14,
  },
  primaryButtonText: {
    color: THEME.surface,
    fontSize: 16,
    fontWeight: "800",
  },
  authPrimaryButton: {
    alignItems: "center",
    backgroundColor: THEME.accent,
    borderRadius: 28,
    marginTop: 22,
    minHeight: 56,
    justifyContent: "center",
  },
  authPrimaryButtonText: {
    color: THEME.surface,
    fontSize: 16,
    fontWeight: "800",
  },
  googleButton: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: "#D9DEE2",
    borderRadius: 28,
    borderWidth: 1,
    justifyContent: "center",
    marginTop: 18,
    minHeight: 56,
  },
  googleButtonText: {
    color: THEME.ink,
    fontSize: 16,
    fontWeight: "800",
  },
  disabled: {
    opacity: 0.65,
  },
  stickySubmitBar: {
    backgroundColor: THEME.background,
    bottom: 76,
    left: 0,
    padding: 18,
    paddingTop: 2,
    position: "absolute",
    right: 0,
  },
  secondaryButton: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 26,
    elevation: 2,
    flex: 1,
    minHeight: 66,
    justifyContent: "center",
    paddingVertical: 10,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.06,
    shadowRadius: 14,
  },
  secondaryButtonActive: {
    backgroundColor: THEME.softAccent,
  },
  secondaryButtonIcon: {
    color: THEME.ink,
    fontSize: 22,
    fontWeight: "900",
    lineHeight: 24,
    marginBottom: 3,
  },
  secondaryButtonText: {
    color: THEME.ink,
    fontSize: 14,
    fontWeight: "800",
  },
  secondaryButtonTextActive: {
    color: THEME.accent,
  },
  scorePanel: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 30,
    elevation: 3,
    paddingHorizontal: 22,
    paddingVertical: 26,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
  },
  scoreLabel: {
    color: THEME.ink,
    fontSize: 22,
    fontWeight: "900",
  },
  scoreValue: {
    fontSize: 58,
    fontWeight: "900",
    marginTop: 8,
  },
  resultFeedbackMessage: {
    color: THEME.slate,
    fontSize: 16,
    fontWeight: "800",
    lineHeight: 23,
    marginTop: 14,
    textAlign: "center",
  },
  resultFeedbackWarning: {
    color: THEME.danger,
  },
  matchStatusCard: {
    backgroundColor: THEME.surface,
    borderRadius: 24,
    elevation: 2,
    marginTop: 14,
    paddingHorizontal: 20,
    paddingVertical: 18,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  matchStatusLabel: {
    color: THEME.slate,
    fontSize: 13,
    fontWeight: "900",
    textTransform: "uppercase",
  },
  matchStatusText: {
    color: THEME.danger,
    fontSize: 18,
    fontWeight: "900",
    lineHeight: 25,
    marginTop: 6,
  },
  resultMeta: {
    color: THEME.ink,
    fontWeight: "700",
  },
  warning: {
    color: THEME.danger,
    fontWeight: "700",
    marginTop: 8,
    textAlign: "center",
  },
  problemText: {
    color: THEME.ink,
    lineHeight: 22,
  },
  explainText: {
    color: THEME.slate,
    lineHeight: 21,
    marginBottom: 12,
  },
  pipelinePanel: {
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    padding: 12,
  },
  pipelineImage: {
    alignSelf: "center",
    backgroundColor: THEME.softMuted,
    borderRadius: 6,
    height: 160,
    marginBottom: 10,
    width: 160,
  },
  resultActions: {
    flexDirection: "row",
    gap: 10,
    marginTop: 18,
  },
  feedbackSummaryCard: {
    backgroundColor: THEME.surface,
    borderRadius: 26,
    elevation: 2,
    flexDirection: "row",
    marginTop: 14,
    paddingHorizontal: 14,
    paddingVertical: 18,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  feedbackSummaryItem: {
    flex: 1,
    minWidth: 0,
    paddingHorizontal: 6,
  },
  feedbackSummaryDivider: {
    backgroundColor: THEME.softMuted,
    width: 1,
  },
  feedbackSummaryLabel: {
    color: THEME.slate,
    fontSize: 11,
    fontWeight: "900",
    textAlign: "center",
    textTransform: "uppercase",
  },
  feedbackSummaryValue: {
    color: THEME.ink,
    fontSize: 15,
    fontWeight: "900",
    lineHeight: 20,
    marginTop: 8,
    textAlign: "center",
  },
  nextStepCard: {
    backgroundColor: THEME.accent,
    borderRadius: 26,
    elevation: 2,
    marginTop: 14,
    paddingHorizontal: 22,
    paddingVertical: 20,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.08,
    shadowRadius: 14,
  },
  nextStepTitle: {
    color: THEME.surface,
    fontSize: 14,
    fontWeight: "900",
    textTransform: "uppercase",
  },
  nextStepText: {
    color: THEME.surface,
    fontSize: 18,
    fontWeight: "900",
    lineHeight: 25,
    marginTop: 8,
  },
  whyResultToggle: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 22,
    elevation: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 14,
    paddingHorizontal: 18,
    paddingVertical: 16,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.05,
    shadowRadius: 12,
  },
  whyResultToggleText: {
    color: THEME.ink,
    fontSize: 15,
    fontWeight: "900",
  },
  whyResultToggleIcon: {
    color: THEME.accent,
    fontSize: 22,
    fontWeight: "900",
  },
  whyResultPanel: {
    backgroundColor: THEME.softAccent,
    borderRadius: 22,
    marginTop: 8,
    paddingHorizontal: 18,
    paddingVertical: 16,
  },
  whyResultText: {
    color: THEME.slate,
    fontSize: 14,
    fontWeight: "700",
    lineHeight: 21,
  },
  comparisonRow: {
    flexDirection: "row",
    gap: 14,
  },
  comparisonPanel: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 26,
    elevation: 2,
    flex: 1,
    padding: 14,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  comparisonLabel: {
    color: THEME.ink,
    fontSize: 14,
    fontWeight: "900",
    marginBottom: 10,
    textAlign: "center",
  },
  comparisonImage: {
    aspectRatio: 1,
    backgroundColor: THEME.softMuted,
    borderRadius: 18,
    width: "100%",
  },
  comparisonPlaceholder: {
    alignItems: "center",
    aspectRatio: 1,
    backgroundColor: THEME.softMuted,
    borderRadius: 18,
    justifyContent: "center",
    width: "100%",
  },
  historyIntroCard: {
    backgroundColor: THEME.accent,
    borderRadius: 30,
    elevation: 3,
    marginBottom: 18,
    paddingHorizontal: 22,
    paddingVertical: 22,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
  },
  historyIntroTitle: {
    color: THEME.surface,
    fontSize: 22,
    fontWeight: "900",
  },
  historyIntroText: {
    color: THEME.surface,
    fontSize: 14,
    fontWeight: "700",
    lineHeight: 21,
    marginTop: 8,
    opacity: 0.92,
  },
  progressRow: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 24,
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 12,
    minHeight: 82,
    paddingHorizontal: 22,
    paddingVertical: 16,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.07,
    shadowRadius: 16,
    elevation: 2,
  },
  progressRowLarge: {
    padding: 18,
  },
  progressName: {
    color: THEME.ink,
    fontSize: 24,
    fontWeight: "900",
  },
  progressMeta: {
    color: THEME.slate,
    fontSize: 13,
    marginTop: 4,
  },
  progressScore: {
    fontSize: 17,
    fontWeight: "900",
  },
  progressNewBadge: {
    alignItems: "center",
    backgroundColor: THEME.softAccent,
    borderRadius: 18,
    minWidth: 58,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  progressNewBadgeText: {
    color: THEME.accent,
    fontSize: 13,
    fontWeight: "800",
  },
  attemptCard: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 28,
    elevation: 2,
    flexDirection: "row",
    marginBottom: 14,
    minHeight: 112,
    padding: 16,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  attemptThumb: {
    backgroundColor: THEME.softMuted,
    borderRadius: 22,
    height: 82,
    width: 82,
  },
  attemptBody: {
    flex: 1,
    marginLeft: 16,
  },
  attemptTitle: {
    color: THEME.ink,
    fontSize: 24,
    fontWeight: "900",
  },
  attemptMeta: {
    color: THEME.slate,
    fontSize: 13,
    fontWeight: "700",
    marginTop: 4,
  },
  attemptScore: {
    fontSize: 22,
    fontWeight: "900",
    marginTop: 8,
  },
  attemptRegion: {
    color: THEME.slate,
    fontSize: 13,
    fontWeight: "800",
    lineHeight: 18,
    marginTop: 4,
  },
  characterDetailHero: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 30,
    elevation: 3,
    marginBottom: 18,
    minHeight: 260,
    padding: 22,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
  },
  characterDetailGlyph: {
    height: 170,
    width: "92%",
  },
  characterDetailDevanagari: {
    color: THEME.ink,
    fontSize: 46,
    fontWeight: "900",
    marginTop: 10,
  },
  profileHeader: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 30,
    flexDirection: "row",
    marginBottom: 32,
    minHeight: 104,
    padding: 20,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
    elevation: 3,
  },
  profileTopBar: {
    alignItems: "center",
    flexDirection: "row",
    height: 48,
    justifyContent: "space-between",
    marginBottom: 30,
  },
  profileIconButton: {
    alignItems: "center",
    height: 48,
    justifyContent: "center",
    width: 48,
  },
  profileBackIcon: {
    height: 24,
    position: "relative",
    width: 24,
  },
  profileBackIconLine: {
    backgroundColor: THEME.ink,
    borderRadius: 2,
    height: 3,
    left: 4,
    position: "absolute",
    width: 15,
  },
  profileBackIconLineTop: {
    top: 6,
    transform: [{ rotate: "-45deg" }],
  },
  profileBackIconLineBottom: {
    bottom: 6,
    transform: [{ rotate: "45deg" }],
  },
  profileGear: {
    color: THEME.ink,
    fontSize: 24,
    fontWeight: "800",
    lineHeight: 48,
    textAlign: "center",
  },
  profileScreenTitle: {
    color: THEME.ink,
    fontSize: 20,
    fontWeight: "900",
    lineHeight: 48,
    textAlign: "center",
  },
  profileAvatar: {
    alignItems: "center",
    backgroundColor: THEME.accent,
    borderRadius: 22,
    height: 64,
    justifyContent: "center",
    width: 64,
  },
  profileAvatarText: {
    color: THEME.surface,
    fontSize: 28,
    fontWeight: "900",
  },
  profileIdentity: {
    flex: 1,
    marginLeft: 16,
  },
  profileName: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "900",
  },
  profileEmail: {
    color: THEME.slate,
    marginTop: 3,
  },
  profileStatsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 16,
  },
  insightStatsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    rowGap: 16,
  },
  insightStatBox: {
    backgroundColor: THEME.surface,
    borderRadius: 28,
    elevation: 2,
    minHeight: 112,
    paddingHorizontal: 22,
    paddingVertical: 22,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
    width: "47.5%",
  },
  insightStatValue: {
    color: THEME.ink,
    fontSize: 38,
    fontWeight: "900",
  },
  insightStatLabel: {
    color: THEME.ink,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0,
    marginTop: 8,
    textTransform: "uppercase",
  },
  profileStatBox: {
    backgroundColor: THEME.surface,
    borderRadius: 28,
    elevation: 2,
    flexBasis: "46%",
    flexGrow: 1,
    minHeight: 100,
    paddingHorizontal: 22,
    paddingVertical: 22,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  profileStatValue: {
    color: THEME.ink,
    fontSize: 30,
    fontWeight: "900",
    lineHeight: 34,
  },
  profileStatLabel: {
    color: THEME.ink,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0,
    marginTop: 4,
    textTransform: "uppercase",
  },
  profileScoreRow: {
    backgroundColor: "#E1E2E5",
    borderRadius: 28,
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 32,
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  profileScoreLabel: {
    color: THEME.ink,
    fontSize: 11,
    fontWeight: "800",
    textTransform: "uppercase",
  },
  profileScoreValue: {
    color: THEME.ink,
    fontSize: 20,
    fontWeight: "900",
    marginTop: 12,
  },
  insightScoreRow: {
    backgroundColor: "#E1E2E5",
    borderRadius: 28,
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 26,
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  insightScoreLabel: {
    color: THEME.ink,
    fontSize: 10,
    fontWeight: "800",
    textTransform: "uppercase",
  },
  insightScoreValue: {
    color: THEME.ink,
    fontSize: 20,
    fontWeight: "900",
    marginTop: 12,
  },
  heatmapPanel: {
    backgroundColor: THEME.surface,
    borderRadius: 28,
    padding: 20,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.07,
    shadowRadius: 18,
    elevation: 3,
  },
  profileSectionTitle: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "900",
    marginBottom: 16,
    marginTop: 34,
  },
  heatmapGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 4,
  },
  heatmapCell: {
    borderRadius: 3,
    height: 13,
    width: 13,
  },
  heatmapLegend: {
    alignItems: "center",
    flexDirection: "row",
    gap: 5,
    justifyContent: "flex-end",
    marginTop: 12,
  },
  heatmapLegendCell: {
    borderRadius: 3,
    height: 11,
    width: 11,
  },
  heatmapLegendText: {
    color: THEME.slate,
    fontSize: 11,
    fontWeight: "700",
  },
  accountPanel: {
    backgroundColor: THEME.surface,
    borderRadius: 28,
    elevation: 2,
    marginBottom: 18,
    padding: 20,
    shadowColor: THEME.slate,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  accountLabel: {
    color: THEME.slate,
    fontSize: 12,
    fontWeight: "900",
    marginBottom: 8,
    marginTop: 12,
    textTransform: "uppercase",
  },
  accountInput: {
    backgroundColor: THEME.background,
    borderRadius: 26,
    color: THEME.ink,
    fontSize: 16,
    fontWeight: "700",
    minHeight: 52,
    paddingHorizontal: 18,
  },
  accountPrimaryButton: {
    alignItems: "center",
    backgroundColor: THEME.accent,
    borderRadius: 26,
    justifyContent: "center",
    marginTop: 20,
    minHeight: 52,
  },
  accountPrimaryButtonText: {
    color: THEME.surface,
    fontSize: 15,
    fontWeight: "900",
  },
  accountSecondaryButton: {
    alignItems: "center",
    backgroundColor: THEME.softAccent,
    borderRadius: 26,
    justifyContent: "center",
    marginTop: 20,
    minHeight: 52,
  },
  accountSecondaryButtonText: {
    color: THEME.accent,
    fontSize: 15,
    fontWeight: "900",
  },
  profileDeactivateButton: {
    alignItems: "center",
    backgroundColor: "#F9E7E4",
    borderRadius: 28,
    justifyContent: "center",
    marginTop: 4,
    minHeight: 56,
  },
  profileDeactivateButtonText: {
    color: THEME.danger,
    fontSize: 16,
    fontWeight: "900",
  },
  profileLogoutButton: {
    alignItems: "center",
    backgroundColor: THEME.ink,
    borderRadius: 28,
    justifyContent: "center",
    marginTop: 28,
    minHeight: 56,
  },
  profileLogoutButtonText: {
    color: THEME.surface,
    fontSize: 16,
    fontWeight: "900",
  },
  bottomNavShell: {
    bottom: 0,
    left: 0,
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 12,
    position: "absolute",
    right: 0,
  },
  bottomNav: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 30,
    elevation: 14,
    flexDirection: "row",
    justifyContent: "space-between",
    minHeight: 72,
    paddingHorizontal: 14,
    shadowColor: THEME.ink,
    shadowOffset: { width: 0, height: -8 },
    shadowOpacity: 0.18,
    shadowRadius: 22,
  },
  bottomNavItem: {
    alignItems: "center",
    borderRadius: 24,
    flex: 1,
    justifyContent: "center",
    minHeight: 54,
    paddingHorizontal: 4,
    paddingVertical: 7,
  },
  bottomNavItemActive: {
    backgroundColor: THEME.accent,
  },
  bottomNavIcon: {
    color: THEME.ink,
    fontSize: 20,
    fontWeight: "900",
    lineHeight: 22,
  },
  bottomNavIconActive: {
    color: THEME.surface,
  },
  bottomNavLabel: {
    color: THEME.ink,
    fontSize: 11,
    fontWeight: "700",
    marginTop: 2,
  },
  bottomNavLabelActive: {
    color: THEME.surface,
  },
  loadingOverlay: {
    alignItems: "center",
    backgroundColor: "rgba(57, 61, 63, 0.45)",
    bottom: 0,
    justifyContent: "center",
    left: 0,
    position: "absolute",
    right: 0,
    top: 0,
  },
  loadingCard: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderRadius: 30,
    elevation: 4,
    paddingHorizontal: 28,
    paddingVertical: 26,
    shadowColor: THEME.ink,
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.18,
    shadowRadius: 24,
    width: "78%",
  },
  loadingTitle: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "900",
    marginTop: 14,
  },
  loadingText: {
    color: THEME.slate,
    fontSize: 14,
    fontWeight: "700",
    lineHeight: 20,
    marginTop: 6,
    textAlign: "center",
  },
  message: {
    backgroundColor: THEME.ink,
    borderRadius: 8,
    bottom: 24,
    left: 18,
    padding: 14,
    position: "absolute",
    right: 18,
  },
  messageText: {
    color: THEME.surface,
    fontWeight: "700",
    textAlign: "center",
  },
});
