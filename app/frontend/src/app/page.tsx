"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload,
  Type,
  Square,
  SquareMinus,
  Trash2,
  Loader2,
  CheckCircle2,
  XCircle,
  Sparkles,
  BoxSelect,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { SegmentationCanvas } from "@/components/SegmentationCanvas";
import {
  uploadImage,
  segmentWithText,
  addBoxPrompt,
  resetPrompts,
  checkHealth,
  type SegmentationResult,
} from "@/lib/api";

type BoxMode = "positive" | "negative";

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [textPrompt, setTextPrompt] = useState("");
  const [boxMode, setBoxMode] = useState<BoxMode>("positive");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check backend health on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const health = await checkHealth();
        setBackendStatus(health.model_loaded ? "online" : "checking");
      } catch {
        setBackendStatus("offline");
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select an image file");
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      // Create preview URL
      const url = URL.createObjectURL(file);
      setImageUrl(url);

      // Upload to backend
      const response = await uploadImage(file);
      setSessionId(response.session_id);
      setImageWidth(response.width);
      setImageHeight(response.height);
      setResult(null);
      setTextPrompt("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload image");
      setImageUrl(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sessionId || !textPrompt.trim()) return;

    setError(null);
    setIsLoading(true);

    try {
      const response = await segmentWithText(sessionId, textPrompt.trim());
      setResult(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Segmentation failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleBoxDrawn = useCallback(
    async (box: number[]) => {
      if (!sessionId) return;

      setError(null);
      setIsLoading(true);

      try {
        const response = await addBoxPrompt(sessionId, box, boxMode === "positive");
        setResult(response.results);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to add box prompt");
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, boxMode]
  );

  const handleReset = async () => {
    if (!sessionId) return;

    setError(null);
    setIsLoading(true);

    try {
      const response = await resetPrompts(sessionId);
      setResult(response.results);
      setTextPrompt("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset");
    } finally {
      setIsLoading(false);
    }
  };

  const maskCount = result?.masks?.length ?? 0;

  return (
    <main className="min-h-screen p-6 md:p-8">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/20 rounded-lg pulse-glow">
              <Sparkles className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">SAM3 Studio</h1>
              <p className="text-sm text-muted-foreground">
                Interactive segmentation with text & box prompts
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {backendStatus === "checking" && (
              <div className="flex items-center gap-2 text-muted-foreground text-sm">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Connecting...</span>
              </div>
            )}
            {backendStatus === "online" && (
              <div className="flex items-center gap-2 text-primary text-sm">
                <CheckCircle2 className="w-4 h-4" />
                <span>Model Ready</span>
              </div>
            )}
            {backendStatus === "offline" && (
              <div className="flex items-center gap-2 text-destructive text-sm">
                <XCircle className="w-4 h-4" />
                <span>Backend Offline</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6">
        {/* Sidebar Controls */}
        <aside className="space-y-4">
          {/* Upload Card */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Upload className="w-4 h-4" />
                Image Source
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                className="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-primary/50 hover:bg-primary/5 transition-all cursor-pointer"
              >
                <Upload className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  Click or drop image here
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileSelect(file);
                  }}
                  className="hidden"
                />
              </div>
              {imageWidth > 0 && (
                <p className="text-xs text-muted-foreground mt-2 text-center">
                  {imageWidth} × {imageHeight} px
                </p>
              )}
            </CardContent>
          </Card>

          {/* Text Prompt Card */}
          <Card className={!sessionId ? "opacity-50 pointer-events-none" : ""}>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Type className="w-4 h-4" />
                Text Prompt
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleTextSubmit} className="flex gap-2">
                <Input
                  value={textPrompt}
                  onChange={(e) => setTextPrompt(e.target.value)}
                  placeholder='e.g. "person", "dog"'
                  disabled={isLoading}
                />
                <Button type="submit" disabled={isLoading || !textPrompt.trim()}>
                  {isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Box Prompt Card */}
          <Card className={!sessionId ? "opacity-50 pointer-events-none" : ""}>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <BoxSelect className="w-4 h-4" />
                Box Prompts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Draw boxes on the image to include or exclude regions
              </p>
              <div className="flex gap-2">
                <Button
                  variant={boxMode === "positive" ? "default" : "secondary"}
                  size="sm"
                  onClick={() => setBoxMode("positive")}
                  className="flex-1"
                >
                  <Square className="w-4 h-4 mr-1" />
                  Include
                </Button>
                <Button
                  variant={boxMode === "negative" ? "destructive" : "secondary"}
                  size="sm"
                  onClick={() => setBoxMode("negative")}
                  className="flex-1"
                >
                  <SquareMinus className="w-4 h-4 mr-1" />
                  Exclude
                </Button>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div
                  className={`w-3 h-3 rounded border-2 ${
                    boxMode === "positive"
                      ? "border-primary bg-primary/20"
                      : "border-muted"
                  }`}
                />
                <span className="text-muted-foreground">
                  Drawing: {boxMode === "positive" ? "Include" : "Exclude"}
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Results & Actions */}
          <Card className={!sessionId ? "opacity-50 pointer-events-none" : ""}>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Objects found:</span>
                <span className="font-medium text-primary">{maskCount}</span>
              </div>
              {result?.prompted_boxes && result.prompted_boxes.length > 0 && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Box prompts:</span>
                  <span className="font-medium">{result.prompted_boxes.length}</span>
                </div>
              )}
              <Button
                variant="destructive"
                size="sm"
                onClick={handleReset}
                disabled={isLoading}
                className="w-full"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear All Prompts
              </Button>
            </CardContent>
          </Card>

          {/* Error Display */}
          {error && (
            <Card className="border-destructive bg-destructive/10">
              <CardContent className="py-3">
                <p className="text-sm text-destructive">{error}</p>
              </CardContent>
            </Card>
          )}
        </aside>

        {/* Main Canvas Area */}
        <section>
          <Card className="overflow-hidden">
            <CardContent className="p-4">
              <SegmentationCanvas
                imageUrl={imageUrl}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                result={result}
                boxMode={boxMode}
                onBoxDrawn={handleBoxDrawn}
                isLoading={isLoading}
              />
            </CardContent>
          </Card>

          {/* Keyboard Shortcuts */}
          {sessionId && (
            <div className="mt-4 flex flex-wrap gap-4 text-xs text-muted-foreground animate-fade-in-up">
              <div className="flex items-center gap-2">
                <kbd className="px-2 py-1 bg-card rounded border border-border font-mono">
                  Click + Drag
                </kbd>
                <span>Draw box</span>
              </div>
              <div className="flex items-center gap-2">
                <kbd className="px-2 py-1 bg-card rounded border border-border font-mono">
                  Enter
                </kbd>
                <span>Submit text prompt</span>
              </div>
            </div>
          )}
        </section>
      </div>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto mt-12 pt-6 border-t border-border">
        <p className="text-xs text-muted-foreground text-center">
          SAM3 Interactive Segmentation • MLX Backend • Next.js Frontend
        </p>
      </footer>
    </main>
  );
}

