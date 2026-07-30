import type {
  Character,
  PracticeAttemptResponse,
  PracticeMode,
  ProgressDashboardItem,
  SelectedImage,
  TokenResponse,
  User,
} from "./types";

export const DEFAULT_API_BASE_URL = "http://192.168.8.100:8000";

type ApiOptions = {
  token?: string | null;
  body?: unknown;
  headers?: Record<string, string>;
};

async function parseApiResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const detail = data?.detail;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
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
