import { forwardRef, useImperativeHandle, useMemo, useRef, useState } from "react";
import { PanResponder, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import type { DimensionValue, ViewStyle } from "react-native";
import Svg, { Path } from "react-native-svg";
import { captureRef } from "react-native-view-shot";

export type DrawingCanvasHandle = {
  clear: () => void;
  capture: () => Promise<string>;
  hasDrawing: () => boolean;
};

type Stroke = {
  id: number;
  path: string;
  color: string;
  width: number;
};

type DrawingCanvasProps = {
  size?: number | DimensionValue;
};

const UI = {
  ink: "#393D3F",
  paper: "#FDFDFF",
  muted: "#C6C5B9",
  accent: "#62929E",
  slate: "#546A7B",
  softAccent: "#E7F0F2",
  softMuted: "#EFEFEB",
};

const INK_COLOR = UI.ink;
const PAPER_COLOR = UI.paper;
const STROKE_WIDTHS = [6, 10, 14, 18];

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(
  ({ size = "100%" }, ref) => {
    const [strokes, setStrokes] = useState<Stroke[]>([]);
    const [undoneStrokes, setUndoneStrokes] = useState<Stroke[]>([]);
    const [activePath, setActivePath] = useState<string>("");
    const [strokeWidth, setStrokeWidth] = useState(10);
    const [tool, setTool] = useState<"pen" | "eraser">("pen");
    const [canvasLayoutSize, setCanvasLayoutSize] = useState(280);
    const paperRef = useRef<View>(null);

    function pointFromEvent(event: { nativeEvent: { locationX: number; locationY: number } }) {
      const scale = 280 / Math.max(1, canvasLayoutSize);
      const x = Math.max(0, Math.min(280, event.nativeEvent.locationX * scale));
      const y = Math.max(0, Math.min(280, event.nativeEvent.locationY * scale));
      return { x, y };
    }

    const panResponder = useMemo(
      () =>
        PanResponder.create({
          onStartShouldSetPanResponder: () => true,
          onMoveShouldSetPanResponder: () => true,
          onPanResponderGrant: (event) => {
            const { x, y } = pointFromEvent(event);
            setActivePath(`M ${x.toFixed(1)} ${y.toFixed(1)}`);
          },
          onPanResponderMove: (event) => {
            const { x, y } = pointFromEvent(event);
            setActivePath((current) => `${current} L ${x.toFixed(1)} ${y.toFixed(1)}`);
          },
          onPanResponderRelease: () => {
            setActivePath((current) => {
              if (current.length > 0) {
                setStrokes((existing) => [
                  ...existing,
                  {
                    id: Date.now(),
                    path: current,
                    color: tool === "eraser" ? PAPER_COLOR : INK_COLOR,
                    width: tool === "eraser" ? strokeWidth + 8 : strokeWidth,
                  },
                ]);
                setUndoneStrokes([]);
              }
              return "";
            });
          },
        }),
      [canvasLayoutSize, strokeWidth, tool],
    );

    useImperativeHandle(ref, () => ({
      clear: () => {
        setActivePath("");
        setStrokes([]);
        setUndoneStrokes([]);
      },
      capture: async () => {
        if (!paperRef.current) {
          throw new Error("Canvas is not ready.");
        }

        return captureRef(paperRef, {
          format: "png",
          quality: 1,
          result: "tmpfile",
        });
      },
      hasDrawing: () => strokes.length > 0 || activePath.length > 0,
    }));

    function undo() {
      setActivePath("");
      setStrokes((current) => {
        if (current.length === 0) {
          return current;
        }
        const next = current.slice(0, -1);
        const removed = current[current.length - 1];
        setUndoneStrokes((undone) => [removed, ...undone]);
        return next;
      });
    }

    function redo() {
      setActivePath("");
      setUndoneStrokes((current) => {
        if (current.length === 0) {
          return current;
        }
        const [restored, ...remaining] = current;
        setStrokes((existing) => [...existing, restored]);
        return remaining;
      });
    }

    function clear() {
      setActivePath("");
      setStrokes([]);
      setUndoneStrokes([]);
    }

    const canvasSizeStyle: ViewStyle =
      typeof size === "number"
        ? { width: size, height: size }
        : { width: size, aspectRatio: 1 };

    return (
      <View style={styles.container}>
        <View style={styles.toolbar}>
          <View style={styles.strokeOptions}>
            {STROKE_WIDTHS.map((width) => (
              <TouchableOpacity
                key={width}
                style={[styles.strokeButton, strokeWidth === width && styles.strokeButtonActive]}
                onPress={() => {
                  setTool("pen");
                  setStrokeWidth(width);
                }}
              >
                <View style={[styles.strokePreview, { height: width / 2, width: 22 }]} />
              </TouchableOpacity>
            ))}
          </View>
          <View style={styles.toolRow}>
            <TouchableOpacity
              style={[styles.toolButton, tool === "eraser" && styles.toolButtonActive]}
              onPress={() => setTool("eraser")}
            >
              <Text style={[styles.toolButtonText, tool === "eraser" && styles.toolButtonTextActive]}>Eraser</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.toolButton} onPress={undo}>
              <Text style={styles.toolButtonText}>Undo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.toolButton} onPress={redo}>
              <Text style={styles.toolButtonText}>Redo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.toolButton} onPress={clear}>
              <Text style={styles.toolButtonText}>Clear</Text>
            </TouchableOpacity>
          </View>
        </View>
        <View
          style={[styles.canvas, canvasSizeStyle]}
          onLayout={(event) => setCanvasLayoutSize(event.nativeEvent.layout.width)}
          {...panResponder.panHandlers}
        >
          <View ref={paperRef} collapsable={false} style={styles.canvasPaper}>
            <Svg width="100%" height="100%" viewBox="0 0 280 280">
              {strokes.map((stroke) => (
                <Path
                  key={stroke.id}
                  d={stroke.path}
                  stroke={stroke.color}
                  strokeWidth={stroke.width}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                />
              ))}
              {activePath.length > 0 ? (
                <Path
                  d={activePath}
                  stroke={tool === "eraser" ? PAPER_COLOR : INK_COLOR}
                  strokeWidth={tool === "eraser" ? strokeWidth + 8 : strokeWidth}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                />
              ) : null}
            </Svg>
          </View>
        </View>
        <View style={styles.canvasFooter}>
          <Text style={styles.hint}>Draw inside the square, then submit.</Text>
        </View>
      </View>
    );
  },
);

const styles = StyleSheet.create({
  container: {
    width: "100%",
  },
  canvas: {
    backgroundColor: PAPER_COLOR,
    borderColor: UI.ink,
    borderRadius: 8,
    borderWidth: 1,
    overflow: "hidden",
  },
  canvasPaper: {
    backgroundColor: PAPER_COLOR,
    flex: 1,
  },
  toolbar: {
    backgroundColor: UI.paper,
    borderColor: UI.muted,
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 10,
    padding: 10,
  },
  strokeOptions: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 10,
  },
  strokeButton: {
    alignItems: "center",
    backgroundColor: UI.softMuted,
    borderColor: UI.muted,
    borderRadius: 7,
    borderWidth: 1,
    flex: 1,
    height: 34,
    justifyContent: "center",
  },
  strokeButtonActive: {
    backgroundColor: UI.softAccent,
    borderColor: UI.accent,
  },
  strokePreview: {
    backgroundColor: INK_COLOR,
    borderRadius: 12,
  },
  toolRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  toolButton: {
    alignItems: "center",
    backgroundColor: UI.paper,
    borderColor: UI.muted,
    borderRadius: 7,
    borderWidth: 1,
    flexGrow: 1,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  toolButtonActive: {
    backgroundColor: UI.accent,
    borderColor: UI.accent,
  },
  toolButtonText: {
    color: UI.ink,
    fontSize: 12,
    fontWeight: "800",
  },
  toolButtonTextActive: {
    color: UI.paper,
  },
  canvasFooter: {
    alignItems: "center",
    marginTop: 8,
  },
  hint: {
    color: UI.slate,
    fontSize: 12,
  },
});
