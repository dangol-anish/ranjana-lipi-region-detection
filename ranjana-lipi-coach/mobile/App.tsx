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
  fetchCharacters,
  fetchCurrentUser,
  fetchProgress,
  loginUser,
  registerUser,
  submitPracticeAttempt,
} from "./src/api";
import { DrawingCanvas, type DrawingCanvasHandle } from "./src/components/DrawingCanvas";
import { RegionGrid } from "./src/components/RegionGrid";
import type {
  Character,
  PracticeAttemptResponse,
  PracticeMode,
  ProgressDashboardItem,
  RegionFeedback,
  SelectedImage,
  User,
} from "./src/types";

type Screen = "auth" | "home" | "practice" | "results" | "progress";
type AuthMode = "login" | "register";
type InputMode = "gallery" | "camera" | "canvas";

const TOKEN_KEY = "ranjana_lipi_token";
const API_BASE_URL_KEY = "ranjana_lipi_api_base_url";
const VALIDATED_CLASSES = new Set(["aa", "a", "ka", "da", "dda"]);
const PRACTICE_MODES: { value: PracticeMode; label: string }[] = [
  { value: "app_suggested", label: "Suggested" },
  { value: "free_practice", label: "Free" },
  { value: "assessment", label: "Assessment" },
];

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
  const [selectedCharacterId, setSelectedCharacterId] = useState<number | null>(null);
  const [selectedMode, setSelectedMode] = useState<PracticeMode>("app_suggested");
  const [inputMode, setInputMode] = useState<InputMode>("gallery");
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [submittedImage, setSubmittedImage] = useState<SelectedImage | null>(null);
  const [result, setResult] = useState<PracticeAttemptResponse | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const drawingRef = useRef<DrawingCanvasHandle>(null);

  const selectedCharacter = useMemo(
    () => characters.find((character) => character.id === selectedCharacterId) ?? characters[0] ?? null,
    [characters, selectedCharacterId],
  );

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

    const [nextCharacters, nextProgress] = await Promise.all([
      fetchCharacters(apiBaseUrl, token),
      fetchProgress(apiBaseUrl, token),
    ]);
    setCharacters(nextCharacters);
    setProgress(nextProgress);
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
    if (!email.trim() || !password.trim()) {
      setMessage("Email and password are required.");
      return;
    }

    if (authMode === "register" && !displayName.trim()) {
      setMessage("Display name is required for registration.");
      return;
    }

    setLoading(true);
    setMessage(null);
    try {
      const response =
        authMode === "register"
          ? await registerUser(apiBaseUrl, email.trim(), password, displayName.trim())
          : await loginUser(apiBaseUrl, email.trim(), password);
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
        if (!drawingRef.current?.hasDrawing()) {
          throw new Error("Draw the character on the canvas before submitting.");
        }
        const uri = await drawingRef.current.capture();
        image = {
          uri,
          name: `${selectedCharacter.name}_canvas_${Date.now()}.png`,
          type: "image/png",
          source: "canvas",
        };
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
    setSelectedMode(mode);
    setSelectedImage(null);
    setSubmittedImage(null);
    setResult(null);
    setMessage(null);
    setScreen("practice");
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
        <Header title="Practice" userName={user?.display_name ?? user?.email ?? "Student"} onLogout={handleLogout} />

        <Text style={styles.sectionTitle}>Choose Mode</Text>
        <View style={styles.modeGrid}>
          {PRACTICE_MODES.map((mode) => (
            <Pressable key={mode.value} style={styles.modeButton} onPress={() => beginPractice(mode.value)}>
              <Text style={styles.modeTitle}>{mode.label}</Text>
              <Text style={styles.modeText}>
                {mode.value === "assessment"
                  ? "Record a scored attempt."
                  : mode.value === "free_practice"
                    ? "Pick any character."
                    : "Start with the app's character set."}
              </Text>
            </Pressable>
          ))}
        </View>

        <Text style={styles.sectionTitle}>Characters</Text>
        <View style={styles.characterGrid}>
          {characters.map((character) => (
            <TouchableOpacity
              key={character.id}
              style={[styles.characterChip, selectedCharacter?.id === character.id && styles.selectedChip]}
              onPress={() => setSelectedCharacterId(character.id)}
            >
              <Text style={[styles.characterName, selectedCharacter?.id === character.id && styles.selectedChipText]}>
                {character.display_label}
              </Text>
              <Text style={[styles.characterSlug, selectedCharacter?.id === character.id && styles.selectedChipText]}>
                {character.name}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.rowBetween}>
          <Text style={styles.sectionTitle}>Progress</Text>
          <TouchableOpacity onPress={() => setScreen("progress")}>
            <Text style={styles.linkText}>Open dashboard</Text>
          </TouchableOpacity>
        </View>
        {progress.slice(0, 5).map((item) => (
          <ProgressRow key={item.character.id} item={item} />
        ))}
      </ScrollView>
    );
  }

  function renderPractice() {
    return (
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <TopNav title="Practice Attempt" onBack={() => setScreen("home")} />

        <Text style={styles.sectionTitle}>Character</Text>
        <View style={styles.characterGrid}>
          {characters.map((character) => (
            <TouchableOpacity
              key={character.id}
              style={[styles.characterChip, selectedCharacter?.id === character.id && styles.selectedChip]}
              onPress={() => setSelectedCharacterId(character.id)}
            >
              <Text style={[styles.characterName, selectedCharacter?.id === character.id && styles.selectedChipText]}>
                {character.display_label}
              </Text>
              <Text style={[styles.characterSlug, selectedCharacter?.id === character.id && styles.selectedChipText]}>
                {character.name}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <Text style={styles.sectionTitle}>Mode</Text>
        <View style={styles.segmented}>
          {PRACTICE_MODES.map((mode) => (
            <TouchableOpacity
              key={mode.value}
              style={[styles.segmentButton, selectedMode === mode.value && styles.segmentButtonActive]}
              onPress={() => setSelectedMode(mode.value)}
            >
              <Text style={[styles.segmentText, selectedMode === mode.value && styles.segmentTextActive]}>
                {mode.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <Text style={styles.sectionTitle}>Input</Text>
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

        <PrimaryButton disabled={loading} label={loading ? "Analyzing..." : "Submit Attempt"} onPress={submitAttempt} />
      </ScrollView>
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

        <Text style={styles.sectionTitle}>Region Map</Text>
        <RegionGrid
          feedback={feedback}
          rows={selectedCharacter?.region_grid_rows ?? 3}
          cols={selectedCharacter?.region_grid_cols ?? 3}
        />

        <Text style={styles.sectionTitle}>Problem Regions</Text>
        <Text style={styles.problemText}>{problemRegionText(feedback)}</Text>

        <View style={styles.resultActions}>
          <SecondaryButton label="Try Again" onPress={() => setScreen("practice")} />
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
          <ProgressRow key={item.character.id} item={item} large />
        ))}
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
              : renderProgress()}
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

function Header({ title, userName, onLogout }: { title: string; userName: string; onLogout: () => void }) {
  return (
    <View style={styles.header}>
      <View>
        <Text style={styles.screenTitle}>{title}</Text>
        <Text style={styles.subtitle}>Welcome, {userName}</Text>
      </View>
      <TouchableOpacity style={styles.smallButton} onPress={onLogout}>
        <Text style={styles.smallButtonText}>Logout</Text>
      </TouchableOpacity>
    </View>
  );
}

function TopNav({ title, onBack }: { title: string; onBack: () => void }) {
  return (
    <View style={styles.header}>
      <TouchableOpacity style={styles.smallButton} onPress={onBack}>
        <Text style={styles.smallButtonText}>Back</Text>
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

function ProgressRow({ item, large }: { item: ProgressDashboardItem; large?: boolean }) {
  const bestScore = item.progress?.best_score ?? null;
  return (
    <View style={[styles.progressRow, large && styles.progressRowLarge]}>
      <View>
        <Text style={styles.progressName}>{item.character.display_label}</Text>
        <Text style={styles.progressMeta}>
          {item.progress?.attempts_count ?? 0} attempts
          {item.progress?.mastered ? " | mastered" : ""}
        </Text>
      </View>
      <Text style={[styles.progressScore, { color: scoreColor(bestScore) }]}>
        {typeof bestScore === "number" ? `${bestScore.toFixed(1)}%` : "New"}
      </Text>
    </View>
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
  navSpacer: {
    width: 58,
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
  },
  characterChip: {
    backgroundColor: "#ffffff",
    borderColor: "#cfdad5",
    borderRadius: 8,
    borderWidth: 1,
    minWidth: 88,
    paddingHorizontal: 14,
    paddingVertical: 12,
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
