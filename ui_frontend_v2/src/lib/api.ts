export type ApiError = {
  message: string;
  status?: number;
  detail?: unknown;
};

async function parseJsonSafe(resp: Response): Promise<unknown> {
  try {
    return await resp.json();
  } catch {
    return null;
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const resp = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers || {}),
    },
  });
  const data = await parseJsonSafe(resp);
  if (!resp.ok) {
    const detail =
      data && typeof data === "object" && "detail" in data ? (data as any).detail : data;
    const err: ApiError = {
      message:
        (typeof detail === "string" && detail) ||
        resp.statusText ||
        "Request failed",
      status: resp.status,
      detail,
    };
    throw err;
  }
  return data as T;
}

