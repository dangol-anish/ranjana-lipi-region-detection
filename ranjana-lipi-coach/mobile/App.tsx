import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import * as SecureStore from "expo-secure-store";
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
  fetchProgress,
  loginUser,
  registerUser,
  submitPracticeAttempt,
} from "./src/api";
import { DrawingCanvas, type DrawingCanvasHandle } from "./src/components/DrawingCanvas";
import { RegionGrid } from "./src/components/RegionGrid";
import type {
  Attempt,
  Character,
  CharacterProgressDetail,
  PracticeAttemptResponse,
  PracticeMode,
  ProgressDashboardItem,
  RegionFeedback,
  SelectedImage,
  User,
  UserProfile,
} from "./src/types";

type Screen = "auth" | "home" | "practice" | "results" | "progress" | "profile" | "history" | "character_detail";
type AuthMode = "login" | "register";
type InputMode = "gallery" | "camera" | "canvas";
type SuggestedPick = {
  item: ProgressDashboardItem;
  reason: string;
};

const TOKEN_KEY = "ranjana_lipi_token";
const API_BASE_URL_KEY = "ranjana_lipi_api_base_url";
const VALIDATED_CLASSES = new Set(["aa", "a", "ka", "da", "dda"]);
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

function scoreColor(score: number | null | undefined): string {
  if (typeof score !== "number") {
    return "#60736c";
  }
  if (score >= 90) {
    return "#1f7a4d";
  }
  if (score >= 70) {
    return "#a46a16";
  }
  return "#b33b2e";
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
    return "#e7ede9";
  }
  if (attempts === 1) {
    return "#bcd9ca";
  }
  if (attempts === 2) {
    return "#7eb996";
  }
  if (attempts <= 4) {
    return "#3f8a63";
  }
  return "#1f5f40";
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

function attemptCharacterLabel(characters: Character[], attempt: Attempt): string {
  const character = characters.find((item) => item.id === attempt.character_id);
  return character ? `${characterDisplayLabel(character)} ${character.name}` : `Character ${attempt.character_id}`;
}

function topRegionSummary(feedback: RegionFeedback | null | undefined): string {
  if (feedback?.insufficient_input) {
    return "Insufficient input";
  }
  const broad = feedback?.broad_bands as { problem_regions?: Array<{ region?: string }> } | undefined;
  const fine = feedback?.fine_grid as { problem_regions?: Array<{ region?: string }> } | undefined;
  const region = broad?.problem_regions?.[0]?.region ?? fine?.problem_regions?.[0]?.region;
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
  if (!progress || progress.attempts_count === 0 || progress.best_score === null) {
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

function suggestedReason(item: ProgressDashboardItem, intervalHours: number, elapsedHours: number): string {
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

function chooseSuggestedPick(progress: ProgressDashboardItem[]): SuggestedPick | null {
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

function toSelectedImage(asset: ImagePicker.ImagePickerAsset, source: "camera" | "gallery"): SelectedImage {
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
  if (feedback?.insufficient_input) {
    return feedback.message ?? feedback.warning ?? "Insufficient input — please draw the full character.";
  }

  const regions = feedback?.problem_regions;
  if (!Array.isArray(regions) || regions.length === 0) {
    return "No strong flawed region was flagged.";
  }

  return regions
    .map((region) => {
      const label = region.label ?? `row ${region.row + 1}, col ${region.col + 1}`;
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
  const [characterDetail, setCharacterDetail] = useState<CharacterProgressDetail | null>(null);
  const [selectedCharacterId, setSelectedCharacterId] = useState<number | null>(null);
  const [selectedMode, setSelectedMode] = useState<PracticeMode>("app_suggested");
  const [inputMode, setInputMode] = useState<InputMode>("gallery");
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [submittedImage, setSubmittedImage] = useState<SelectedImage | null>(null);
  const [result, setResult] = useState<PracticeAttemptResponse | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [characterSearch, setCharacterSearch] = useState("");
  const [characterPickerOpen, setCharacterPickerOpen] = useState(false);
  const [suggestedReasonText, setSuggestedReasonText] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const drawingRef = useRef<DrawingCanvasHandle>(null);

  const selectedCharacter = useMemo(
    () => characters.find((character) => character.id === selectedCharacterId) ?? characters[0] ?? null,
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

    if (storedBaseUrl) {
      setApiBaseUrl(storedBaseUrl);
    }

    if (!storedToken) {
      return;
    }

    try {
      const baseUrl = storedBaseUrl ?? DEFAULT_API_BASE_URL;
      const currentUser = await fetchCurrentUser(baseUrl, storedToken);
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

    const [nextCharacters, nextProgress, nextProfile, nextAttempts] = await Promise.all([
      fetchCharacters(apiBaseUrl, token),
      fetchProgress(apiBaseUrl, token),
      fetchProfile(apiBaseUrl, token),
      fetchAttemptHistory(apiBaseUrl, token, 50),
    ]);
    setCharacters(nextCharacters);
    setProgress(nextProgress);
    setProfile(nextProfile);
    setAttemptHistory(nextAttempts);
    setSelectedCharacterId((current) => current ?? nextCharacters[0]?.id ?? null);
  }, [apiBaseUrl, token]);

  useEffect(() => {
    void loadSession();
  }, [loadSession]);

  useEffect(() => {
    if (token) {
      void refreshAppData().catch((error: unknown) => {
        setMessage(error instanceof Error ? error.message : "Could not load app data.");
      });
    }
  }, [refreshAppData, token]);

  async function handleAuth() {
    const emailValue = email.trim();
    const displayNameValue = displayName.trim();

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

    if (authMode === "register" && !displayNameValue) {
      setMessage("Display name is required for registration.");
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      const response =
        authMode === "register"
          ? await registerUser(apiBaseUrl, emailValue, password, displayNameValue)
          : await loginUser(apiBaseUrl, emailValue, password);
      await SecureStore.setItemAsync(TOKEN_KEY, response.access_token);
      await SecureStore.setItemAsync(API_BASE_URL_KEY, apiBaseUrl);
      const currentUser = await fetchCurrentUser(apiBaseUrl, response.access_token);
      setToken(response.access_token);
      setUser(currentUser);
      setScreen("home");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Authentication failed.");
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
      Alert.alert("Permission needed", "Gallery access is needed to choose a handwriting sample.");
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
      Alert.alert("Permission needed", "Camera access is needed to photograph a handwriting sample.");
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
          throw new Error("Draw the character on the canvas before submitting.");
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
      const response = await submitPracticeAttempt(apiBaseUrl, token, selectedCharacter.id, selectedMode, image);
      setResult(response);
      await refreshAppData();
      setScreen("results");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not submit attempt.");
    } finally {
      setLoading(false);
    }
  }

  function beginPractice(mode: PracticeMode) {
    if (mode === "app_suggested") {
      const pick = chooseSuggestedPick(progress);
      if (pick) {
        setSelectedCharacterId(pick.item.character.id);
        setSuggestedReasonText(pick.reason);
      } else {
        setSuggestedReasonText(null);
      }
    } else {
      setSuggestedReasonText(null);
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

  function continuePractice() {
    if (selectedMode === "app_suggested") {
      const pick = chooseSuggestedPick(progress);
      if (pick) {
        setSelectedCharacterId(pick.item.character.id);
        setSuggestedReasonText(pick.reason);
      }
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
      const detail = await fetchCharacterProgress(apiBaseUrl, token, characterId);
      setCharacterDetail(detail);
      setScreen("character_detail");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not load character profile.");
    } finally {
      setLoading(false);
    }
  }

  function renderAuth() {
    return (
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={styles.centerScreen}>
        <View style={styles.authPanel}>
          <Text style={styles.brand}>Ranjana Lipi Learning App</Text>
          <Text style={styles.subtitle}>Sign in to save attempts and progress.</Text>

          <Text style={styles.label}>Backend URL</Text>
          <TextInput
            autoCapitalize="none"
            autoCorrect={false}
            style={styles.input}
            value={apiBaseUrl}
            onChangeText={setApiBaseUrl}
            placeholder="http://192.168.x.x:8000"
          />
          <Text style={styles.fieldHint}>
            Expo Go on a phone must use this Mac's Wi-Fi IP, not 127.0.0.1.
          </Text>

          <Text style={styles.label}>Email</Text>
          <TextInput
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="email-address"
            style={styles.input}
            value={email}
            onChangeText={setEmail}
          />

          <Text style={styles.label}>Password</Text>
          <TextInput secureTextEntry style={styles.input} value={password} onChangeText={setPassword} />

          {authMode === "register" ? (
            <>
              <Text style={styles.label}>Display name</Text>
              <TextInput style={styles.input} value={displayName} onChangeText={setDisplayName} />
            </>
          ) : null}

          <PrimaryButton
            disabled={loading}
            label={loading ? "Please wait..." : authMode === "login" ? "Log In" : "Create Account"}
            onPress={handleAuth}
          />
          <TouchableOpacity
            style={styles.linkButton}
            onPress={() => {
              setAuthMode(authMode === "login" ? "register" : "login");
              setMessage(null);
            }}
          >
            <Text style={styles.linkText}>
              {authMode === "login" ? "Create a new account" : "Already have an account? Log in"}
            </Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    );
  }

  function renderHome() {
    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Header
          title="Ranjana Lipi Learning App"
          userName={user?.display_name ?? user?.email ?? "Student"}
          onLogout={handleLogout}
          onHistory={() => setScreen("history")}
          onProfile={() => setScreen("profile")}
        />

        <Text style={styles.sectionTitle}>Learn at your pace</Text>
        <View style={styles.modeGrid}>
          {PRACTICE_MODES.map((mode) => (
            <Pressable key={mode.value} style={styles.modeButton} onPress={() => beginPractice(mode.value)}>
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
          <ProgressRow key={item.character.id} item={item} onPress={() => void openCharacterDetail(item.character.id)} />
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
        <ScrollView contentContainerStyle={styles.practiceScrollContent} keyboardShouldPersistTaps="handled">
          <TopNav title={practiceTitle} onBack={() => setScreen("home")} />

          {selectedCharacter ? (
            <View style={styles.practiceGlyphBox}>
              <Image
                source={{ uri: characterGlyphUri(apiBaseUrl, selectedCharacter.name) }}
                style={styles.practiceGlyphImage}
                resizeMode="contain"
              />
              <Text style={styles.practiceDevanagariLabel}>{characterDisplayLabel(selectedCharacter)}</Text>
            </View>
          ) : null}

          <Text style={styles.sectionTitle}>Character</Text>
          <View style={styles.selectedCharacterPanel}>
            <View>
              <Text style={styles.selectedCharacterLabel}>Selected</Text>
              <Text style={styles.selectedCharacterName}>{selectedCharacter?.name ?? "No character"}</Text>
              {selectedMode === "app_suggested" && suggestedReasonText ? (
                <Text style={styles.suggestedReasonText}>{suggestedReasonText}</Text>
              ) : null}
            </View>
            {selectedMode === "free_practice" ? (
              <TouchableOpacity
                style={styles.changeCharacterButton}
                onPress={() => setCharacterPickerOpen((current) => !current)}
              >
                <Text style={styles.changeCharacterText}>{characterPickerOpen ? "Done" : "Change"}</Text>
              </TouchableOpacity>
            ) : null}
          </View>

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
                    style={[styles.characterChip, selectedCharacter?.id === character.id && styles.selectedChip]}
                    onPress={() => {
                      setSelectedCharacterId(character.id);
                      setCharacterPickerOpen(false);
                      setCharacterSearch("");
                    }}
                  >
                    <Text
                      style={[styles.characterName, selectedCharacter?.id === character.id && styles.selectedChipText]}
                    >
                      {characterDisplayLabel(character)}
                    </Text>
                    <Text
                      style={[styles.characterSlug, selectedCharacter?.id === character.id && styles.selectedChipText]}
                    >
                      {character.name}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
              {filteredCharacters.length === 0 ? (
                <Text style={styles.emptyText}>No characters match that search.</Text>
              ) : null}
            </View>
          ) : null}

          <Text style={styles.sectionTitle}>How would you like to input?</Text>
          <View style={styles.inputActions}>
            <SecondaryButton label="Gallery" onPress={pickFromGallery} active={inputMode === "gallery"} />
            <SecondaryButton label="Camera" onPress={takePhoto} active={inputMode === "camera"} />
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
                  <Text style={styles.demoCanvasButtonText}>Use Demo Image</Text>
                </TouchableOpacity>
              ) : null}
              {selectedImage?.source === "demo_canvas" ? (
                <View style={styles.previewWrap}>
                  <Image source={{ uri: selectedImage.uri }} style={styles.previewImage} resizeMode="contain" />
                  <Text style={styles.previewText}>Demo canvas image selected.</Text>
                </View>
              ) : null}
              <DrawingCanvas ref={drawingRef} />
            </View>
          ) : selectedImage ? (
            <View style={styles.previewWrap}>
              <Image source={{ uri: selectedImage.uri }} style={styles.previewImage} resizeMode="contain" />
              <Text style={styles.previewText}>{selectedImage.name}</Text>
            </View>
          ) : (
            <Text style={styles.emptyText}>Choose a photo from gallery or camera.</Text>
          )}
        </ScrollView>

        <View style={styles.stickySubmitBar}>
          <PrimaryButton disabled={loading} label={loading ? "Analyzing..." : "Submit Attempt"} onPress={submitAttempt} />
        </View>
      </View>
    );
  }

  function renderResults() {
    const feedback = result?.region_feedback ?? null;
    const score = result?.overall_score ?? feedback?.overall_score ?? null;
    const referenceUri = selectedCharacter
      ? `${apiBaseUrl}/${VALIDATED_CLASSES.has(selectedCharacter.name) ? "references" : "references_general"}/${selectedCharacter.name}.png`
      : null;

    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Feedback" onBack={() => setScreen("practice")} />
        <View style={styles.scorePanel}>
          <Text style={styles.scoreLabel}>Overall Score</Text>
          <Text style={[styles.scoreValue, { color: scoreColor(score) }]}>
            {typeof score === "number" ? `${score.toFixed(1)}%` : "--"}
          </Text>
        </View>

        <Text style={styles.sectionTitle}>Comparison</Text>
        <View style={styles.comparisonRow}>
          <View style={styles.comparisonPanel}>
            <Text style={styles.comparisonLabel}>Reference</Text>
            {referenceUri ? (
              <Image source={{ uri: referenceUri }} style={styles.comparisonImage} resizeMode="contain" />
            ) : (
              <View style={styles.comparisonPlaceholder}>
                <Text style={styles.emptyText}>No reference</Text>
              </View>
            )}
          </View>
          <View style={styles.comparisonPanel}>
            <Text style={styles.comparisonLabel}>Your Input</Text>
            {submittedImage ? (
              <Image source={{ uri: submittedImage.uri }} style={styles.comparisonImage} resizeMode="contain" />
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
                The submitted image is binarized, aligned to the selected character reference, and placed on a fixed
                canvas before feedback is calculated.
              </Text>
            </View>
          </>
        ) : null}

        <Text style={styles.sectionTitle}>Region Map</Text>
        <RegionGrid
          feedback={feedback}
          rows={selectedCharacter?.region_grid_rows ?? 3}
          cols={selectedCharacter?.region_grid_cols ?? 3}
        />

        <Text style={styles.sectionTitle}>Problem Regions</Text>
        <Text style={styles.problemText}>{problemRegionText(feedback)}</Text>

        <Text style={styles.sectionTitle}>How This Feedback Was Produced</Text>
        <Text style={styles.explainText}>
          The app compares the normalized attempt with the expected reconstruction for this character. Regions with
          unusually high ink-masked reconstruction error are highlighted as likely places to improve.
        </Text>

        <View style={styles.resultActions}>
          <SecondaryButton label={selectedMode === "app_suggested" ? "Next Suggested" : "Try Again"} onPress={continuePractice} />
          <SecondaryButton label="Progress" onPress={() => setScreen("progress")} />
        </View>
      </ScrollView>
    );
  }

  function renderProgress() {
    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Progress Dashboard" onBack={() => setScreen("home")} />
        {progress.map((item) => (
          <ProgressRow key={item.character.id} item={item} large onPress={() => void openCharacterDetail(item.character.id)} />
        ))}
      </ScrollView>
    );
  }

  function renderAttemptCard(attempt: Attempt) {
    return (
      <View key={attempt.id} style={styles.attemptCard}>
        <Image source={{ uri: attemptImageUri(apiBaseUrl, attempt) }} style={styles.attemptThumb} resizeMode="contain" />
        <View style={styles.attemptBody}>
          <Text style={styles.attemptTitle}>{attemptCharacterLabel(characters, attempt)}</Text>
          <Text style={styles.attemptMeta}>{formatAttemptDate(attempt.created_at)} | {attempt.mode.replace("_", " ")}</Text>
          <Text style={[styles.attemptScore, { color: scoreColor(attempt.overall_score) }]}>
            {scoreText(attempt.overall_score)}
          </Text>
          <Text style={styles.attemptRegion}>{topRegionSummary(attempt.region_feedback)}</Text>
        </View>
      </View>
    );
  }

  function renderHistory() {
    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Attempt History" onBack={() => setScreen("profile")} />
        <Text style={styles.explainText}>
          Saved attempts show the normalized handwriting image, score, and strongest region feedback from the same
          reconstruction-based pipeline used during practice.
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
          <TopNav title="Character Profile" onBack={() => setScreen("progress")} />
          <Text style={styles.emptyText}>No character profile loaded.</Text>
        </ScrollView>
      );
    }

    const progressItem: ProgressDashboardItem = {
      character: detail.character,
      progress: detail.progress,
    };
    const intervalHours = ankiStyleIntervalHours(progressItem);
    const elapsedHours = hoursSince(detail.progress?.last_practiced_at, Date.now());
    const reviewStatus = suggestedReason(progressItem, intervalHours, elapsedHours);

    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Character Profile" onBack={() => setScreen("progress")} />
        <View style={styles.characterDetailHero}>
          <Image
            source={{ uri: characterGlyphUri(apiBaseUrl, detail.character.name) }}
            style={styles.characterDetailGlyph}
            resizeMode="contain"
          />
          <Text style={styles.characterDetailDevanagari}>{characterDisplayLabel(detail.character)}</Text>
          <Text style={styles.characterDetailSlug}>{detail.character.name}</Text>
        </View>

        <View style={styles.profileStatsGrid}>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>{detail.progress?.attempts_count ?? 0}</Text>
            <Text style={styles.profileStatLabel}>Attempts</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>{scoreText(detail.progress?.best_score)}</Text>
            <Text style={styles.profileStatLabel}>Best Score</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>{detail.progress?.mastered ? "Yes" : "No"}</Text>
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
          detail.attempts.slice(0, 10).map((attempt) => renderAttemptCard(attempt))
        ) : (
          <Text style={styles.emptyText}>No attempts for this character yet.</Text>
        )}
      </ScrollView>
    );
  }

  function renderProfile() {
    const stats = profile?.stats;
    const displayName = profile?.user.display_name ?? profile?.user.email ?? user?.display_name ?? user?.email ?? "Student";
    const heatmap = profile?.heatmap ?? [];

    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Profile" onBack={() => setScreen("home")} />

        <View style={styles.profileHeader}>
          <View style={styles.profileAvatar}>
            <Text style={styles.profileAvatarText}>{displayName.slice(0, 1).toUpperCase()}</Text>
          </View>
          <View style={styles.profileIdentity}>
            <Text style={styles.profileName}>{displayName}</Text>
            <Text style={styles.profileEmail}>{profile?.user.email ?? user?.email}</Text>
          </View>
        </View>

        <View style={styles.profileStatsGrid}>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>{stats?.total_attempts ?? 0}</Text>
            <Text style={styles.profileStatLabel}>Attempts</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>{stats?.practiced_characters ?? 0}</Text>
            <Text style={styles.profileStatLabel}>Practiced</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>{stats?.mastered_characters ?? 0}</Text>
            <Text style={styles.profileStatLabel}>Mastered</Text>
          </View>
          <View style={styles.profileStatBox}>
            <Text style={styles.profileStatValue}>{stats?.current_streak_days ?? 0}</Text>
            <Text style={styles.profileStatLabel}>Day Streak</Text>
          </View>
        </View>

        <View style={styles.profileScoreRow}>
          <View>
            <Text style={styles.profileScoreLabel}>Average Score</Text>
            <Text style={[styles.profileScoreValue, { color: scoreColor(stats?.average_score) }]}>
              {scoreText(stats?.average_score)}
            </Text>
          </View>
          <View>
            <Text style={styles.profileScoreLabel}>Best Score</Text>
            <Text style={[styles.profileScoreValue, { color: scoreColor(stats?.best_score) }]}>
              {scoreText(stats?.best_score)}
            </Text>
          </View>
          <View>
            <Text style={styles.profileScoreLabel}>Longest Streak</Text>
            <Text style={styles.profileScoreValue}>{stats?.longest_streak_days ?? 0}d</Text>
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
              <View key={count} style={[styles.heatmapLegendCell, { backgroundColor: heatmapColor(count) }]} />
            ))}
            <Text style={styles.heatmapLegendText}>More</Text>
          </View>
        </View>

        <PrimaryButton label="View Attempt History" onPress={() => setScreen("history")} />
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
          <ActivityIndicator color="#ffffff" />
        </View>
      ) : null}
      {message ? (
        <TouchableOpacity style={styles.message} onPress={() => setMessage(null)}>
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
      <View>
        <Text style={styles.screenTitle}>{title}</Text>
        <Text style={styles.subtitle}>Welcome, {userName}</Text>
      </View>
      <View style={styles.headerMenu}>
        <TouchableOpacity
          accessibilityLabel="Open account menu"
          style={styles.iconButton}
          onPress={() => setMenuOpen((current) => !current)}
        >
          <Text style={styles.iconButtonText}>⋮</Text>
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
    <View style={styles.header}>
      <TouchableOpacity accessibilityLabel="Go back" style={styles.backIconButton} onPress={onBack}>
        <Text style={styles.backIconText}>‹</Text>
      </TouchableOpacity>
      <Text style={styles.screenTitle}>{title}</Text>
      <View style={styles.navSpacer} />
    </View>
  );
}

function PrimaryButton({ label, onPress, disabled }: { label: string; onPress: () => void; disabled?: boolean }) {
  return (
    <TouchableOpacity disabled={disabled} style={[styles.primaryButton, disabled && styles.disabled]} onPress={onPress}>
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
    <TouchableOpacity style={[styles.secondaryButton, active && styles.secondaryButtonActive]} onPress={onPress}>
      <Text style={[styles.secondaryButtonText, active && styles.secondaryButtonTextActive]}>{label}</Text>
    </TouchableOpacity>
  );
}

function ProgressRow({ item, large, onPress }: { item: ProgressDashboardItem; large?: boolean; onPress?: () => void }) {
  const bestScore = item.progress?.best_score ?? null;
  const Container = onPress ? TouchableOpacity : View;
  return (
    <Container style={[styles.progressRow, large && styles.progressRowLarge]} onPress={onPress}>
      <View>
        <Text style={styles.progressName}>{characterDisplayLabel(item.character)}</Text>
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
    backgroundColor: "#f3f6f2",
  },
  centerScreen: {
    flex: 1,
    justifyContent: "center",
    padding: 20,
  },
  authPanel: {
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    padding: 20,
  },
  brand: {
    color: "#17211e",
    fontSize: 30,
    fontWeight: "800",
    marginBottom: 6,
  },
  subtitle: {
    color: "#66736f",
    fontSize: 14,
  },
  label: {
    color: "#263238",
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 6,
    marginTop: 14,
  },
  input: {
    backgroundColor: "#f8faf7",
    borderColor: "#cfdad5",
    borderRadius: 8,
    borderWidth: 1,
    color: "#17211e",
    minHeight: 46,
    paddingHorizontal: 12,
  },
  fieldHint: {
    color: "#66736f",
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
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 22,
  },
  screenTitle: {
    color: "#17211e",
    fontSize: 26,
    fontWeight: "800",
  },
  smallButton: {
    borderColor: "#b8c6c1",
    borderRadius: 7,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  smallButtonText: {
    color: "#263238",
    fontWeight: "700",
  },
  backIconButton: {
    alignItems: "center",
    height: 42,
    justifyContent: "center",
    width: 42,
  },
  backIconText: {
    color: "#263238",
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
    height: 38,
    justifyContent: "center",
    width: 34,
  },
  iconButtonText: {
    color: "#263238",
    fontSize: 28,
    fontWeight: "900",
    lineHeight: 30,
  },
  menuPopover: {
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    minWidth: 120,
    position: "absolute",
    right: 0,
    top: 44,
    zIndex: 20,
  },
  menuItem: {
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  menuItemText: {
    color: "#263238",
    fontWeight: "800",
  },
  navSpacer: {
    width: 42,
  },
  sectionTitle: {
    color: "#263238",
    fontSize: 18,
    fontWeight: "800",
    marginBottom: 12,
    marginTop: 12,
  },
  modeGrid: {
    gap: 10,
  },
  modeButton: {
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    padding: 16,
  },
  modeTitle: {
    color: "#17211e",
    fontSize: 18,
    fontWeight: "800",
  },
  modeText: {
    color: "#66736f",
    marginTop: 4,
  },
  characterGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 12,
  },
  characterChip: {
    backgroundColor: "#ffffff",
    borderColor: "#cfdad5",
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
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
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
    color: "#17211e",
    fontSize: 32,
    fontWeight: "900",
    marginTop: 8,
  },
  selectedCharacterPanel: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    padding: 14,
  },
  selectedCharacterLabel: {
    color: "#66736f",
    fontSize: 12,
    fontWeight: "800",
  },
  selectedCharacterName: {
    color: "#17211e",
    fontSize: 18,
    fontWeight: "900",
    marginTop: 2,
  },
  suggestedReasonText: {
    color: "#66736f",
    fontSize: 12,
    fontWeight: "700",
    marginTop: 3,
  },
  changeCharacterButton: {
    backgroundColor: "#21443a",
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 9,
  },
  changeCharacterText: {
    color: "#ffffff",
    fontWeight: "800",
  },
  characterPickerPanel: {
    marginTop: 10,
  },
  selectedChip: {
    backgroundColor: "#21443a",
    borderColor: "#21443a",
  },
  characterName: {
    color: "#17211e",
    fontSize: 18,
    fontWeight: "800",
  },
  characterSlug: {
    color: "#66736f",
    fontSize: 12,
    marginTop: 2,
  },
  selectedChipText: {
    color: "#ffffff",
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
    color: "#246b55",
    fontWeight: "800",
  },
  segmented: {
    backgroundColor: "#e4ebe7",
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
    backgroundColor: "#ffffff",
  },
  segmentText: {
    color: "#60736c",
    fontSize: 12,
    fontWeight: "800",
  },
  segmentTextActive: {
    color: "#17211e",
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
    backgroundColor: "#21443a",
    borderRadius: 8,
    marginBottom: 10,
    paddingVertical: 12,
    width: "100%",
  },
  demoCanvasButtonText: {
    color: "#ffffff",
    fontWeight: "900",
  },
  previewWrap: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
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
    color: "#66736f",
    fontSize: 12,
    marginTop: 8,
  },
  emptyText: {
    color: "#66736f",
    marginBottom: 18,
  },
  primaryButton: {
    alignItems: "center",
    backgroundColor: "#21443a",
    borderRadius: 8,
    marginTop: 16,
    paddingVertical: 14,
  },
  primaryButtonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "800",
  },
  disabled: {
    opacity: 0.65,
  },
  stickySubmitBar: {
    backgroundColor: "#f3f6f2",
    borderColor: "#d7e0dc",
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
    backgroundColor: "#ffffff",
    borderColor: "#b8c6c1",
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    paddingVertical: 11,
  },
  secondaryButtonActive: {
    backgroundColor: "#dce9e3",
    borderColor: "#246b55",
  },
  secondaryButtonText: {
    color: "#263238",
    fontWeight: "800",
  },
  secondaryButtonTextActive: {
    color: "#21443a",
  },
  scorePanel: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    padding: 18,
  },
  scoreLabel: {
    color: "#66736f",
    fontWeight: "700",
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: "900",
    marginTop: 4,
  },
  resultMeta: {
    color: "#263238",
    fontWeight: "700",
  },
  warning: {
    color: "#a13d2e",
    fontWeight: "700",
    marginTop: 8,
    textAlign: "center",
  },
  problemText: {
    color: "#263238",
    lineHeight: 22,
  },
  explainText: {
    color: "#4f625d",
    lineHeight: 21,
    marginBottom: 12,
  },
  pipelinePanel: {
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    padding: 12,
  },
  pipelineImage: {
    alignSelf: "center",
    backgroundColor: "#f8faf7",
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
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    padding: 10,
  },
  comparisonLabel: {
    color: "#263238",
    fontSize: 12,
    fontWeight: "800",
    marginBottom: 8,
    textAlign: "center",
  },
  comparisonImage: {
    aspectRatio: 1,
    backgroundColor: "#f8faf7",
    borderRadius: 6,
    width: "100%",
  },
  comparisonPlaceholder: {
    alignItems: "center",
    aspectRatio: 1,
    backgroundColor: "#f8faf7",
    borderRadius: 6,
    justifyContent: "center",
    width: "100%",
  },
  progressRow: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
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
    color: "#17211e",
    fontSize: 17,
    fontWeight: "800",
  },
  progressMeta: {
    color: "#66736f",
    marginTop: 3,
  },
  progressScore: {
    fontSize: 18,
    fontWeight: "900",
  },
  attemptCard: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    marginBottom: 10,
    padding: 10,
  },
  attemptThumb: {
    backgroundColor: "#f8faf7",
    borderRadius: 6,
    height: 72,
    width: 72,
  },
  attemptBody: {
    flex: 1,
    marginLeft: 12,
  },
  attemptTitle: {
    color: "#17211e",
    fontSize: 16,
    fontWeight: "900",
  },
  attemptMeta: {
    color: "#66736f",
    fontSize: 12,
    marginTop: 2,
  },
  attemptScore: {
    fontSize: 20,
    fontWeight: "900",
    marginTop: 5,
  },
  attemptRegion: {
    color: "#4f625d",
    fontSize: 12,
    fontWeight: "700",
    marginTop: 3,
  },
  characterDetailHero: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
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
    color: "#17211e",
    fontSize: 34,
    fontWeight: "900",
    marginTop: 6,
  },
  characterDetailSlug: {
    color: "#66736f",
    fontSize: 14,
    fontWeight: "800",
    marginTop: 2,
  },
  profileHeader: {
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    marginBottom: 12,
    padding: 16,
  },
  profileAvatar: {
    alignItems: "center",
    backgroundColor: "#21443a",
    borderRadius: 28,
    height: 56,
    justifyContent: "center",
    width: 56,
  },
  profileAvatarText: {
    color: "#ffffff",
    fontSize: 24,
    fontWeight: "900",
  },
  profileIdentity: {
    flex: 1,
    marginLeft: 12,
  },
  profileName: {
    color: "#17211e",
    fontSize: 20,
    fontWeight: "900",
  },
  profileEmail: {
    color: "#66736f",
    marginTop: 3,
  },
  profileStatsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  profileStatBox: {
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    flexBasis: "47%",
    flexGrow: 1,
    padding: 14,
  },
  profileStatValue: {
    color: "#17211e",
    fontSize: 28,
    fontWeight: "900",
  },
  profileStatLabel: {
    color: "#66736f",
    fontSize: 12,
    fontWeight: "800",
    marginTop: 3,
  },
  profileScoreRow: {
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 12,
    padding: 14,
  },
  profileScoreLabel: {
    color: "#66736f",
    fontSize: 11,
    fontWeight: "800",
  },
  profileScoreValue: {
    color: "#17211e",
    fontSize: 18,
    fontWeight: "900",
    marginTop: 4,
  },
  heatmapPanel: {
    backgroundColor: "#ffffff",
    borderColor: "#d7e0dc",
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
    color: "#66736f",
    fontSize: 11,
    fontWeight: "700",
  },
  loadingOverlay: {
    alignItems: "center",
    backgroundColor: "rgba(23, 33, 30, 0.45)",
    bottom: 0,
    justifyContent: "center",
    left: 0,
    position: "absolute",
    right: 0,
    top: 0,
  },
  message: {
    backgroundColor: "#263238",
    borderRadius: 8,
    bottom: 24,
    left: 18,
    padding: 14,
    position: "absolute",
    right: 18,
  },
  messageText: {
    color: "#ffffff",
    fontWeight: "700",
    textAlign: "center",
  },
});
