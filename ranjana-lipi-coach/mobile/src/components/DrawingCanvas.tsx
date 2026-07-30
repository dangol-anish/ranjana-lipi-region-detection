import { forwardRef, useImperativeHandle, useMemo, useRef, useState } from "react";
import { PanResponder, StyleSheet, Text, TouchableOpacity, View } from "react-native";
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
};

type DrawingCanvasProps = {
  size?: number;
};

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(
  ({ size = 280 }, ref) => {
    const [strokes, setStrokes] = useState<Stroke[]>([]);
    const [activePath, setActivePath] = useState<string>("");
    const canvasRef = useRef<View>(null);

    const panResponder = useMemo(
      () =>
        PanResponder.create({
          onStartShouldSetPanResponder: () => true,
          onMoveShouldSetPanResponder: () => true,
          onPanResponderGrant: (event) => {
            const { locationX, locationY } = event.nativeEvent;
            setActivePath(`M ${locationX.toFixed(1)} ${locationY.toFixed(1)}`);
          },
          onPanResponderMove: (event) => {
            const { locationX, locationY } = event.nativeEvent;
            setActivePath((current) => `${current} L ${locationX.toFixed(1)} ${locationY.toFixed(1)}`);
          },
          onPanResponderRelease: () => {
            setActivePath((current) => {
              if (current.length > 0) {
                setStrokes((existing) => [...existing, { id: Date.now(), path: current }]);
              }
              return "";
            });
          },
        }),
      [],
    );

    useImperativeHandle(ref, () => ({
      clear: () => {
        setActivePath("");
        setStrokes([]);
      },
      capture: async () => {
        if (!canvasRef.current) {
          throw new Error("Canvas is not ready.");
        }

        return captureRef(canvasRef, {
          format: "png",
          quality: 1,
          result: "tmpfile",
        });
      },
      hasDrawing: () => strokes.length > 0 || activePath.length > 0,
    }));

    return (
      <View>
        <View
          ref={canvasRef}
          collapsable={false}
          style={[styles.canvas, { width: size, height: size }]}
          {...panResponder.panHandlers}
        >
          <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
            {strokes.map((stroke) => (
              <Path
                key={stroke.id}
                d={stroke.path}
                stroke="#111816"
                strokeWidth={12}
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            ))}
            {activePath.length > 0 ? (
              <Path
                d={activePath}
                stroke="#111816"
                strokeWidth={12}
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            ) : null}
          </Svg>
        </View>
        <View style={styles.canvasFooter}>
          <Text style={styles.hint}>Draw inside the square, then submit.</Text>
          <TouchableOpacity
            style={styles.clearButton}
            onPress={() => {
              setActivePath("");
              setStrokes([]);
            }}
          >
            <Text style={styles.clearButtonText}>Clear</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  },
);

const styles = StyleSheet.create({
  canvas: {
    backgroundColor: "#fffefb",
    borderColor: "#263238",
    borderRadius: 8,
    borderWidth: 1,
    overflow: "hidden",
  },
  canvasFooter: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 8,
  },
  hint: {
    color: "#65736f",
    fontSize: 12,
  },
  clearButton: {
    borderColor: "#aab7b2",
    borderRadius: 6,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  clearButtonText: {
    color: "#263238",
    fontWeight: "700",
  },
});
