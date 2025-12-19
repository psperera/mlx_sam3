const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface UploadResponse {
  session_id: string;
  width: number;
  height: number;
  message: string;
}

export interface SegmentationResult {
  original_width: number;
  original_height: number;
  masks?: number[][][][]; // [N, 1, H, W] - each mask has 1 channel
  boxes?: number[][];     // [N, 4] as [x0, y0, x1, y1]
  scores?: number[];      // [N]
  prompted_boxes?: { box: number[]; label: boolean }[];
}

export interface SegmentResponse {
  session_id: string;
  prompt?: string;
  box_type?: string;
  results: SegmentationResult;
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to upload image");
  }

  return response.json();
}

export async function segmentWithText(
  sessionId: string,
  prompt: string
): Promise<SegmentResponse> {
  const response = await fetch(`${API_BASE}/segment/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, prompt }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to segment");
  }

  return response.json();
}

export async function addBoxPrompt(
  sessionId: string,
  box: number[],
  label: boolean
): Promise<SegmentResponse> {
  const response = await fetch(`${API_BASE}/segment/box`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, box, label }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to add box prompt");
  }

  return response.json();
}

export async function resetPrompts(sessionId: string): Promise<SegmentResponse> {
  const response = await fetch(`${API_BASE}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to reset");
  }

  return response.json();
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error("Backend not available");
  }
  return response.json();
}

