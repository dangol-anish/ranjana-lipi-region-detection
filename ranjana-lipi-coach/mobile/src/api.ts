import type {
  Character,
  Attempt,
  CharacterProgressDetail,
  PracticeAttemptResponse,
  PracticeMode,
  ProgressDashboardItem,
  SelectedImage,
  TokenResponse,
  User,
  UserProfile,
} from "./types";

export const DEFAULT_API_BASE_URL = "http://192.168.8.100:8000";

type ApiOptions = {
  token?: string | null;
  body?: unknown;
  headers?: Record<string, string>;
};

type ValidationDetail = {
  loc?: Array<string | number>;
  msg?: string;
  type?: string;
};

function friendlyFieldName(field: string): string {
  const names: Record<string, string> = {
    email: "Email",
    password: "Password",
    display_name: "Display name",
    character_id: "Character",
    mode: "Practice mode",
    image: "Image",
  };
  return names[field] ?? field.replaceAll("_", " ");
}

function friendlyValidationMessage(detail: ValidationDetail): string {
  const field = detail.loc?.[detail.loc.length - 1];
  const fieldName = typeof field === "string" ? friendlyFieldName(field) : "This field";
  const type = detail.type ?? "";

  if (field === "password" && type.includes("string_too_short")) {
    return "Password must be at least 8 characters.";
  }
  if (field === "email" && type.includes("string_too_long")) {
    return "Email is too long.";
  }
  if (field === "display_name" && type.includes("string_too_long")) {
    return "Display name is too long.";
  }
  if (type.includes("missing")) {
    return `${fieldName} is required.`;
  }

  return detail.msg ? `${fieldName}: ${detail.msg}` : `${fieldName} is invalid.`;
}

function apiErrorMessage(status: number, data: unknown): string {
  const detail = typeof data === "object" && data !== null && "detail" in data ? (data as { detail?: unknown }).detail : null;

  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail.map((item) => friendlyValidationMessage(item as ValidationDetail)).join("\n");
  }

  return `Request failed (${status}). Please try again.`;
}

async function parseApiResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    throw new Error(apiErrorMessage(response.status, data));
  }

  return data as T;
}

export async function apiRequest<T>(
  baseUrl: string,
  path: string,
  options: ApiOptions = {},
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options.headers ?? {}),
  };

  if (options.token) {
    headers.Authorization = `Bearer ${options.token}`;
  }

  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${baseUrl}${path}`, {
    method: options.body ? "POST" : "GET",
    headers,
    body: options.body instanceof FormData ? options.body : options.body ? JSON.stringify(options.body) : undefined,
  });

  return parseApiResponse<T>(response);
}

export function registerUser(
  baseUrl: string,
  email: string,
  password: string,
  displayName: string,
): Promise<TokenResponse> {
  return apiRequest<TokenResponse>(baseUrl, "/auth/register", {
    body: {
      email,
      password,
      display_name: displayName,
    },
  });
}

export function loginUser(baseUrl: string, email: string, password: string): Promise<TokenResponse> {
  return apiRequest<TokenResponse>(baseUrl, "/auth/login", {
    body: {
      email,
      password,
    },
  });
}

export function fetchCurrentUser(baseUrl: string, token: string): Promise<User> {
  return apiRequest<User>(baseUrl, "/auth/me", { token });
}

export function fetchCharacters(baseUrl: string, token: string): Promise<Character[]> {
  return apiRequest<Character[]>(baseUrl, "/characters", { token });
}

export function fetchProgress(baseUrl: string, token: string): Promise<ProgressDashboardItem[]> {
  return apiRequest<ProgressDashboardItem[]>(baseUrl, "/progress", { token });
}

export function fetchCharacterProgress(
  baseUrl: string,
  token: string,
  characterId: number,
): Promise<CharacterProgressDetail> {
  return apiRequest<CharacterProgressDetail>(baseUrl, `/progress/${characterId}`, { token });
}

export function fetchProfile(baseUrl: string, token: string): Promise<UserProfile> {
  return apiRequest<UserProfile>(baseUrl, "/profile/me", { token });
}

export function fetchAttemptHistory(baseUrl: string, token: string, limit = 50): Promise<Attempt[]> {
  return apiRequest<Attempt[]>(baseUrl, `/practice/attempts?limit=${limit}`, { token });
}

export async function submitPracticeAttempt(
  baseUrl: string,
  token: string,
  characterId: number,
  mode: PracticeMode,
  image: SelectedImage,
): Promise<PracticeAttemptResponse> {
  const form = new FormData();
  form.append("character_id", String(characterId));
  form.append("mode", mode);
  form.append("image", {
    uri: image.uri,
    name: image.name,
    type: image.type,
  } as unknown as Blob);

  const response = await fetch(`${baseUrl}/practice/attempt`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: form,
  });

  return parseApiResponse<PracticeAttemptResponse>(response);
}
