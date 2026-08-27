import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import * as SecureStore from "expo-secure-store";
import { LinearGradient } from "expo-linear-gradient";
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
  da: "द",
  dda: "ड",
  ddha: "ढ",
  dha: "ध",
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
  na: "न",
  nine: "९",
  nna: "ण",
  nnna: "ऩ",
  nya: "ञ",
  o: "ओ",
  one: "१",
  pa: "प",
  pha: "फ",
  ra: "र",
  ri: "ऋ",
  rii: "ॠ",
  sa: "स",
  saa: "ष",
  seven: "७",
  sha: "श",
  six: "६",
  ta: "त",
  tha: "थ",
  three: "३",
  tra: "त्र",
  tta: "ट",
  ttha: "ठ",
  two: "२",
  u: "उ",
  uu: "ऊ",
  wo: "व",
  ya: "य",
  zero: "०",
};

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

function characterGlyphUri(baseUrl: string, characterName: string): string {
  return `${baseUrl}/display_glyphs/${characterName}.png`;
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
    ? `${characterDisplayLabel(character)} ${character.name}`
    : `Character ${attempt.character_id}`;
}

function topRegionSummary(feedback: RegionFeedback | null | undefined): string {
  if (feedback?.wrong_character) {
    return "Wrong character";
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
  const [characterSearch, setCharacterSearch] = useState("");
  const [characterPickerOpen, setCharacterPickerOpen] = useState(false);
  const [suggestedReasonText, setSuggestedReasonText] = useState<string | null>(
    null,
  );
  const [suggestedRecommendation, setSuggestedRecommendation] =
    useState<PracticeRecommendation | null>(null);
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
    if (!query) {
      return characters;
    }
    return characters.filter((character) => {
      return (
        character.name.toLowerCase().includes(query) ||
        character.display_label.toLowerCase().includes(query) ||
        characterDisplayLabel(character).includes(query)
      );
    });
  }, [characterSearch, characters]);

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
    setScreen("auth");
    setResult(null);
    setSelectedImage(null);
    setSubmittedImage(null);
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
    setMessage(null);
    setCharacterSearch("");
    setCharacterPickerOpen(mode === "free_practice");
    setScreen("practice");
  }

  async function continuePractice() {
    if (selectedMode === "app_suggested") {
      await loadSuggestedRecommendation();
    }

    setSelectedImage(null);
    setSubmittedImage(null);
    setResult(null);
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

            <PrimaryButton
              disabled={loading}
              label={
                loading
                  ? "Please wait..."
                  : authMode === "login"
                    ? "Log In"
                    : "Create Account"
              }
              onPress={handleAuth}
            />
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
          userName={user?.display_name ?? user?.email ?? "Student"}
          onLogout={handleLogout}
          onHistory={() => setScreen("history")}
          onProfile={() => setScreen("profile")}
        />

        <Text style={styles.sectionTitle}>Learn at your pace</Text>
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
          <Text style={styles.sectionTitle}>Progress</Text>
          <TouchableOpacity onPress={() => setScreen("progress")}>
            <Text style={styles.linkText}>Open dashboard</Text>
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
            </View>
          ) : null}

          <Text style={styles.sectionTitle}>Character</Text>
          <View style={styles.selectedCharacterPanel}>
            <View>
              <Text style={styles.selectedCharacterLabel}>Selected</Text>
              <Text style={styles.selectedCharacterName}>
                {selectedCharacter?.name ?? "No character"}
              </Text>
              {selectedMode === "app_suggested" && suggestedReasonText ? (
                <Text style={styles.suggestedReasonText}>
                  {suggestedReasonText}
                </Text>
              ) : null}
            </View>
            {selectedMode === "free_practice" ? (
              <TouchableOpacity
                style={styles.changeCharacterButton}
                onPress={() => setCharacterPickerOpen((current) => !current)}
              >
                <Text style={styles.changeCharacterText}>
                  {characterPickerOpen ? "Done" : "Change"}
                </Text>
              </TouchableOpacity>
            ) : null}
          </View>

          {selectedMode === "app_suggested" && suggestedRecommendation ? (
            <View style={styles.recommendationPanel}>
              <Text style={styles.recommendationTitle}>
                Adaptive Recommendation
              </Text>
              <Text style={styles.recommendationText}>
                {suggestedRecommendation.reason}
              </Text>
              <View style={styles.recommendationStats}>
                <View style={styles.recommendationStat}>
                  <Text style={styles.recommendationStatValue}>
                    {suggestedRecommendation.priority_score.toFixed(1)}
                  </Text>
                  <Text style={styles.recommendationStatLabel}>Priority</Text>
                </View>
                <View style={styles.recommendationStat}>
                  <Text style={styles.recommendationStatValue}>
                    {scoreText(
                      suggestedRecommendation.signals.recent_average_score,
                    )}
                  </Text>
                  <Text style={styles.recommendationStatLabel}>Recent Avg</Text>
                </View>
                <View style={styles.recommendationStat}>
                  <Text style={styles.recommendationStatValue}>
                    {suggestedRecommendation.signals.weakest_region ?? "None"}
                  </Text>
                  <Text style={styles.recommendationStatLabel}>
                    Weak Region
                  </Text>
                </View>
              </View>
            </View>
          ) : null}

          {selectedMode === "free_practice" && characterPickerOpen ? (
            <View style={styles.characterPickerPanel}>
              <TextInput
                autoCapitalize="none"
                autoCorrect={false}
                placeholder="Search characters"
                style={styles.input}
                value={characterSearch}
                onChangeText={setCharacterSearch}
              />
              <View style={styles.characterGrid}>
                {filteredCharacters.map((character) => (
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
                    <Text
                      style={[
                        styles.characterSlug,
                        selectedCharacter?.id === character.id &&
                          styles.selectedChipText,
                      ]}
                    >
                      {character.name}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
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
              onPress={pickFromGallery}
              active={inputMode === "gallery"}
            />
            <SecondaryButton
              label="Camera"
              onPress={takePhoto}
              active={inputMode === "camera"}
            />
            <SecondaryButton
              label="Canvas"
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
            <Text style={styles.emptyText}>
              Choose a photo from gallery or camera.
            </Text>
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
    const referenceUri = selectedCharacter
      ? `${apiBaseUrl}/reference_photos/${selectedCharacter.name}/photo-1/${selectedCharacter.name}.jpg`
      : null;

    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Feedback" onBack={() => setScreen("practice")} />
        <View style={styles.scorePanel}>
          <Text style={styles.scoreLabel}>
            {isWrongCharacter ? "Character Match" : "Overall Score"}
          </Text>
          <Text style={[styles.scoreValue, { color: scoreColor(score) }]}>
            {typeof score === "number" ? `${score.toFixed(1)}%` : "--"}
          </Text>
          {isWrongCharacter ? (
            <Text style={styles.warning}>
              {feedback?.warning ?? feedback?.message}
            </Text>
          ) : null}
        </View>

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
            <Text style={styles.comparisonLabel}>Your Input</Text>
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

        {result?.attempt ? (
          <>
            <Text style={styles.sectionTitle}>Normalized Input</Text>
            <View style={styles.pipelinePanel}>
              <Image
                source={{ uri: attemptImageUri(apiBaseUrl, result.attempt) }}
                style={styles.pipelineImage}
                resizeMode="contain"
              />
              <Text style={styles.explainText}>
                The submitted image is binarized, aligned to the selected
                character reference, and placed on a fixed canvas before
                feedback is calculated.
              </Text>
            </View>
          </>
        ) : null}

        <Text style={styles.sectionTitle}>Region Map</Text>
        <RegionGrid feedback={feedback} />

        <Text style={styles.sectionTitle}>Problem Regions</Text>
        <Text style={styles.problemText}>{problemRegionText(feedback)}</Text>

        <Text style={styles.sectionTitle}>How This Feedback Was Produced</Text>
        <Text style={styles.explainText}>
          {isWrongCharacter
            ? "The recognizer detected that this attempt does not match the selected character, so scoring was blocked instead of showing a misleading reconstruction score."
            : feedback?.feedback_method === "structural_part_mask"
              ? "The app checks whether the normalized attempt covers required structural parts of the taught character form. Missing required parts are shown as top, middle, or bottom feedback."
              : feedback?.feedback_method === "statistical_template"
                ? "The app compares the normalized attempt with a statistical handwriting envelope learned from correct samples for this character. Missing required stroke zones and extra ink outside the allowed variation zone are highlighted."
                : "The app compares the normalized attempt with the expected reconstruction for this character. Regions with unusually high ink-masked reconstruction error are highlighted as likely places to improve."}
        </Text>

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
    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Progress Dashboard" onBack={() => setScreen("home")} />
        {progress.map((item) => (
          <ProgressRow
            key={item.character.id}
            item={item}
            large
            onPress={() => void openCharacterDetail(item.character.id)}
          />
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
        <Text style={styles.explainText}>
          Saved attempts show the normalized handwriting image, score, and
          strongest region feedback from the same reconstruction-based pipeline
          used during practice.
        </Text>
        {attemptHistory.length > 0 ? (
          attemptHistory.map((attempt) => renderAttemptCard(attempt))
        ) : (
          <Text style={styles.emptyText}>No attempts yet.</Text>
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
          <Text style={styles.emptyText}>No character profile loaded.</Text>
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
          <Text style={styles.characterDetailSlug}>
            {detail.character.name}
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
          <Text style={styles.emptyText}>
            No attempts for this character yet.
          </Text>
        )}
      </ScrollView>
    );
  }

  function renderProfile() {
    const stats = profile?.stats;
    const displayName =
      profile?.user.display_name ??
      profile?.user.email ??
      user?.display_name ??
      user?.email ??
      "Student";
    const heatmap = profile?.heatmap ?? [];

    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Profile" onBack={() => setScreen("home")} />

        <View style={styles.profileHeader}>
          <View style={styles.profileAvatar}>
            <Text style={styles.profileAvatarText}>
              {displayName.slice(0, 1).toUpperCase()}
            </Text>
          </View>
          <View style={styles.profileIdentity}>
            <Text style={styles.profileName}>{displayName}</Text>
            <Text style={styles.profileEmail}>
              {profile?.user.email ?? user?.email}
            </Text>
          </View>
        </View>

        <View style={styles.profileStatsGrid}>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>
              {stats?.total_attempts ?? 0}
            </Text>
            <Text style={styles.profileStatLabel}>Attempts</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>
              {stats?.practiced_characters ?? 0}
            </Text>
            <Text style={styles.profileStatLabel}>Practiced</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>
              {stats?.mastered_characters ?? 0}
            </Text>
            <Text style={styles.profileStatLabel}>Mastered</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>
              {stats?.current_streak_days ?? 0}
            </Text>
            <Text style={styles.profileStatLabel}>Day Streak</Text>
          </View>
        </View>

        <View style={styles.profileScoreRow}>
          <View>
            <Text style={styles.profileScoreLabel}>Average Score</Text>
            <Text
              style={[
                styles.profileScoreValue,
                { color: scoreColor(stats?.average_score) },
              ]}
            >
              {scoreText(stats?.average_score)}
            </Text>
          </View>
          <View>
            <Text style={styles.profileScoreLabel}>Best Score</Text>
            <Text
              style={[
                styles.profileScoreValue,
                { color: scoreColor(stats?.best_score) },
              ]}
            >
              {scoreText(stats?.best_score)}
            </Text>
          </View>
          <View>
            <Text style={styles.profileScoreLabel}>Longest Streak</Text>
            <Text style={styles.profileScoreValue}>
              {stats?.longest_streak_days ?? 0}d
            </Text>
          </View>
        </View>

        <Text style={styles.sectionTitle}>Practice Heatmap</Text>
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

        <PrimaryButton
          label="View Attempt History"
          onPress={() => setScreen("history")}
        />
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
      {loading && screen !== "auth" ? (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator color={THEME.surface} />
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

function Header({
  title,
  userName,
  onLogout,
  onHistory,
  onProfile,
}: {
  title: string;
  userName: string;
  onLogout: () => void;
  onHistory: () => void;
  onProfile: () => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);

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
      <View style={styles.headerMenu}>
        <TouchableOpacity
          accessibilityLabel="Open account menu"
          style={styles.iconButton}
          onPress={() => setMenuOpen((current) => !current)}
        >
          <View style={styles.menuDot} />
          <View style={styles.menuDot} />
          <View style={styles.menuDot} />
        </TouchableOpacity>
        {menuOpen ? (
          <View style={styles.menuPopover}>
            <TouchableOpacity
              style={styles.menuItem}
              onPress={() => {
                setMenuOpen(false);
                onProfile();
              }}
            >
              <Text style={styles.menuItemText}>Profile</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.menuItem}
              onPress={() => {
                setMenuOpen(false);
                onHistory();
              }}
            >
              <Text style={styles.menuItemText}>Attempt History</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.menuItem}
              onPress={() => {
                setMenuOpen(false);
                onLogout();
              }}
            >
              <Text style={styles.menuItemText}>Logout</Text>
            </TouchableOpacity>
          </View>
        ) : null}
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
        <Text style={styles.backIconText}>‹</Text>
      </TouchableOpacity>
      <Text style={styles.screenTitle}>{title}</Text>
      <View style={styles.navSpacer} />
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
  onPress,
  active,
}: {
  label: string;
  onPress: () => void;
  active?: boolean;
}) {
  return (
    <TouchableOpacity
      style={[styles.secondaryButton, active && styles.secondaryButtonActive]}
      onPress={onPress}
    >
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
      <Text style={[styles.progressScore, { color: scoreColor(bestScore) }]}>
        {typeof bestScore === "number" ? `${bestScore.toFixed(1)}%` : "New"}
      </Text>
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
    marginTop: 14,
  },
  fieldHint: {
    color: THEME.slate,
    fontSize: 12,
    marginTop: 6,
  },
  scrollContent: {
    padding: 18,
    paddingBottom: 40,
    paddingTop: 56,
  },
  practiceScreen: {
    flex: 1,
  },
  practiceScrollContent: {
    padding: 18,
    paddingBottom: 120,
    paddingTop: 56,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  topNavHeader: {
    alignItems: "center",
  },
  headerBrand: {
    alignItems: "flex-start",
    flex: 1,
    paddingRight: 14,
  },
  screenTitle: {
    color: THEME.ink,
    fontSize: 26,
    fontWeight: "800",
  },
  headerLogo: {
    height: 30,
    marginLeft: -30,
    width: 146,
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
    height: 42,
    justifyContent: "center",
    width: 42,
  },
  backIconText: {
    color: THEME.ink,
    fontSize: 40,
    fontWeight: "700",
    lineHeight: 40,
  },
  headerMenu: {
    alignItems: "flex-end",
    position: "relative",
  },
  iconButton: {
    alignItems: "center",
    borderRadius: 18,
    gap: 3,
    height: 36,
    justifyContent: "center",
    marginTop: 1,
    width: 36,
  },
  menuDot: {
    backgroundColor: THEME.ink,
    borderRadius: 2,
    height: 4,
    width: 4,
  },
  menuPopover: {
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    minWidth: 156,
    position: "absolute",
    right: 0,
    top: 42,
    zIndex: 20,
  },
  menuItem: {
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  menuItemText: {
    color: THEME.ink,
    fontWeight: "800",
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
  modeGrid: {
    gap: 10,
  },
  modeButton: {
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    padding: 16,
  },
  modeTitle: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "800",
  },
  modeText: {
    color: THEME.slate,
    marginTop: 4,
  },
  characterGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 12,
  },
  characterChip: {
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flexBasis: "30%",
    flexGrow: 1,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  practiceGlyphBox: {
    alignItems: "center",
    alignSelf: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    height: 220,
    justifyContent: "center",
    marginBottom: 10,
    marginTop: 8,
    width: "72%",
  },
  practiceGlyphImage: {
    height: 150,
    width: "88%",
  },
  practiceDevanagariLabel: {
    color: THEME.ink,
    fontSize: 32,
    fontWeight: "900",
    marginTop: 8,
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
  changeCharacterButton: {
    backgroundColor: THEME.accent,
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 9,
  },
  changeCharacterText: {
    color: THEME.surface,
    fontWeight: "800",
  },
  characterPickerPanel: {
    marginTop: 10,
  },
  selectedChip: {
    backgroundColor: THEME.accent,
    borderColor: THEME.accent,
  },
  characterName: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "800",
  },
  characterSlug: {
    color: THEME.slate,
    fontSize: 12,
    marginTop: 2,
  },
  selectedChipText: {
    color: THEME.surface,
  },
  rowBetween: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 10,
  },
  linkButton: {
    alignItems: "center",
    marginTop: 14,
  },
  linkText: {
    color: THEME.accent,
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
    gap: 8,
    marginBottom: 16,
  },
  canvasWrap: {
    alignItems: "center",
    marginBottom: 18,
    width: "100%",
  },
  demoCanvasButton: {
    alignItems: "center",
    backgroundColor: THEME.accent,
    borderRadius: 8,
    marginBottom: 10,
    paddingVertical: 12,
    width: "100%",
  },
  demoCanvasButtonText: {
    color: THEME.surface,
    fontWeight: "900",
  },
  previewWrap: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 18,
    padding: 12,
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
  primaryButton: {
    alignItems: "center",
    backgroundColor: THEME.accent,
    borderRadius: 8,
    marginTop: 16,
    paddingVertical: 14,
  },
  primaryButtonText: {
    color: THEME.surface,
    fontSize: 16,
    fontWeight: "800",
  },
  googleButton: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.ink,
    borderRadius: 8,
    borderWidth: 1.5,
    marginTop: 12,
    paddingVertical: 14,
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
    borderColor: THEME.muted,
    borderTopWidth: 1,
    bottom: 0,
    left: 0,
    padding: 18,
    paddingTop: 2,
    position: "absolute",
    right: 0,
  },
  secondaryButton: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    paddingVertical: 11,
  },
  secondaryButtonActive: {
    backgroundColor: THEME.softAccent,
    borderColor: THEME.accent,
  },
  secondaryButtonText: {
    color: THEME.ink,
    fontWeight: "800",
  },
  secondaryButtonTextActive: {
    color: THEME.accent,
  },
  scorePanel: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    padding: 18,
  },
  scoreLabel: {
    color: THEME.slate,
    fontWeight: "700",
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: "900",
    marginTop: 4,
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
  comparisonRow: {
    flexDirection: "row",
    gap: 10,
  },
  comparisonPanel: {
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    padding: 10,
  },
  comparisonLabel: {
    color: THEME.ink,
    fontSize: 12,
    fontWeight: "800",
    marginBottom: 8,
    textAlign: "center",
  },
  comparisonImage: {
    aspectRatio: 1,
    backgroundColor: THEME.softMuted,
    borderRadius: 6,
    width: "100%",
  },
  comparisonPlaceholder: {
    alignItems: "center",
    aspectRatio: 1,
    backgroundColor: THEME.softMuted,
    borderRadius: 6,
    justifyContent: "center",
    width: "100%",
  },
  progressRow: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 10,
    padding: 14,
  },
  progressRowLarge: {
    padding: 18,
  },
  progressName: {
    color: THEME.ink,
    fontSize: 17,
    fontWeight: "800",
  },
  progressMeta: {
    color: THEME.slate,
    marginTop: 3,
  },
  progressScore: {
    fontSize: 18,
    fontWeight: "900",
  },
  attemptCard: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    marginBottom: 10,
    padding: 10,
  },
  attemptThumb: {
    backgroundColor: THEME.softMuted,
    borderRadius: 6,
    height: 72,
    width: 72,
  },
  attemptBody: {
    flex: 1,
    marginLeft: 12,
  },
  attemptTitle: {
    color: THEME.ink,
    fontSize: 16,
    fontWeight: "900",
  },
  attemptMeta: {
    color: THEME.slate,
    fontSize: 12,
    marginTop: 2,
  },
  attemptScore: {
    fontSize: 20,
    fontWeight: "900",
    marginTop: 5,
  },
  attemptRegion: {
    color: THEME.slate,
    fontSize: 12,
    fontWeight: "700",
    marginTop: 3,
  },
  characterDetailHero: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 12,
    padding: 16,
  },
  characterDetailGlyph: {
    height: 140,
    width: "80%",
  },
  characterDetailDevanagari: {
    color: THEME.ink,
    fontSize: 34,
    fontWeight: "900",
    marginTop: 6,
  },
  characterDetailSlug: {
    color: THEME.slate,
    fontSize: 14,
    fontWeight: "800",
    marginTop: 2,
  },
  profileHeader: {
    alignItems: "center",
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    marginBottom: 12,
    padding: 16,
  },
  profileAvatar: {
    alignItems: "center",
    backgroundColor: THEME.accent,
    borderRadius: 28,
    height: 56,
    justifyContent: "center",
    width: 56,
  },
  profileAvatarText: {
    color: THEME.surface,
    fontSize: 24,
    fontWeight: "900",
  },
  profileIdentity: {
    flex: 1,
    marginLeft: 12,
  },
  profileName: {
    color: THEME.ink,
    fontSize: 20,
    fontWeight: "900",
  },
  profileEmail: {
    color: THEME.slate,
    marginTop: 3,
  },
  profileStatsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  profileStatBox: {
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flexBasis: "47%",
    flexGrow: 1,
    padding: 14,
  },
  profileStatValue: {
    color: THEME.ink,
    fontSize: 28,
    fontWeight: "900",
  },
  profileStatLabel: {
    color: THEME.slate,
    fontSize: 12,
    fontWeight: "800",
    marginTop: 3,
  },
  profileScoreRow: {
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 12,
    padding: 14,
  },
  profileScoreLabel: {
    color: THEME.slate,
    fontSize: 11,
    fontWeight: "800",
  },
  profileScoreValue: {
    color: THEME.ink,
    fontSize: 18,
    fontWeight: "900",
    marginTop: 4,
  },
  heatmapPanel: {
    backgroundColor: THEME.surface,
    borderColor: THEME.muted,
    borderRadius: 8,
    borderWidth: 1,
    padding: 12,
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
