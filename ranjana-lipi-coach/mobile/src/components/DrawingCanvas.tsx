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

type ToolIconName = "pen" | "eraser" | "undo" | "redo" | "clear";

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

function ToolIcon({ name, active = false }: { name: ToolIconName; active?: boolean }) {
  const color = active ? UI.paper : UI.ink;
  const common = {
    fill: "none",
    stroke: color,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    strokeWidth: 2.4,
  };

  const paths: Record<ToolIconName, string[]> = {
    pen: [
      "M4 20l4.5-1 10.7-10.7a2.2 2.2 0 0 0 0-3.1l-.4-.4a2.2 2.2 0 0 0-3.1 0L5 15.5 4 20z",
      "M13.8 5.7l4.5 4.5",
    ],
    eraser: [
      "M4 15.5l8.9-8.9a2.2 2.2 0 0 1 3.1 0l2.4 2.4a2.2 2.2 0 0 1 0 3.1L11.5 19H4v-3.5z",
      "M9.5 10.5l5 5",
      "M13 19h7",
    ],
    undo: [
      "M9 7H4v5",
      "M4.8 11.2A7.5 7.5 0 1 0 7 6.6",
    ],
    redo: [
      "M15 7h5v5",
      "M19.2 11.2A7.5 7.5 0 1 1 17 6.6",
    ],
    clear: [
      "M5 7h14",
      "M9 7V5h6v2",
      "M7 10l1 10h8l1-10",
      "M10.5 13v4",
      "M13.5 13v4",
    ],
  };

  return (
    <Svg width={24} height={24} viewBox="0 0 24 24">
      {paths[name].map((path) => (
        <Path key={path} d={path} {...common} />
      ))}
    </Svg>
  );
}

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
          <Text style={styles.toolbarLabel}>Stroke width</Text>
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
          <Text style={styles.toolbarLabel}>Tools</Text>
          <View style={styles.toolRow}>
            <TouchableOpacity
              accessibilityLabel="Pen"
              style={[styles.toolButton, tool === "pen" && styles.toolButtonActive]}
              onPress={() => setTool("pen")}
            >
              <ToolIcon name="pen" active={tool === "pen"} />
            </TouchableOpacity>
            <TouchableOpacity
              accessibilityLabel="Eraser"
              style={[styles.toolButton, tool === "eraser" && styles.toolButtonActive]}
              onPress={() => setTool("eraser")}
            >
              <ToolIcon name="eraser" active={tool === "eraser"} />
            </TouchableOpacity>
            <TouchableOpacity accessibilityLabel="Undo" style={styles.toolButton} onPress={undo}>
              <ToolIcon name="undo" />
            </TouchableOpacity>
            <TouchableOpacity accessibilityLabel="Redo" style={styles.toolButton} onPress={redo}>
              <ToolIcon name="redo" />
            </TouchableOpacity>
            <TouchableOpacity accessibilityLabel="Clear" style={styles.toolButton} onPress={clear}>
              <ToolIcon name="clear" />
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
    backgroundColor: UI.paper,
    borderRadius: 30,
    elevation: 3,
    padding: 16,
    shadowColor: UI.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
    width: "100%",
  },
  canvas: {
    backgroundColor: PAPER_COLOR,
    borderColor: UI.softMuted,
    borderRadius: 24,
    borderWidth: 1,
    overflow: "hidden",
  },
  canvasPaper: {
    backgroundColor: PAPER_COLOR,
    flex: 1,
  },
  toolbar: {
    backgroundColor: UI.softMuted,
    borderRadius: 24,
    marginBottom: 14,
    padding: 12,
  },
  toolbarLabel: {
    color: UI.ink,
    fontSize: 12,
    fontWeight: "900",
    marginBottom: 8,
    textTransform: "uppercase",
  },
  strokeOptions: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 14,
  },
  strokeButton: {
    alignItems: "center",
    backgroundColor: UI.paper,
    borderRadius: 18,
    flex: 1,
    height: 42,
    justifyContent: "center",
  },
  strokeButtonActive: {
    backgroundColor: UI.accent,
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
    borderRadius: 20,
    flexGrow: 1,
    minHeight: 42,
    paddingHorizontal: 10,
    paddingVertical: 10,
  },
  toolButtonActive: {
    backgroundColor: UI.accent,
  },
  canvasFooter: {
    alignItems: "center",
    marginTop: 12,
  },
  hint: {
    color: UI.slate,
    fontSize: 13,
    fontWeight: "700",
  },
});
