import { StyleSheet, Text, View } from "react-native";

import type { RegionFeedback, RegionScore } from "../types";

const CELL_LABELS = [
  "top-left",
  "top-center",
  "top-right",
  "middle-left",
  "middle-center",
  "middle-right",
  "bottom-left",
  "bottom-center",
  "bottom-right",
];

function regionKey(region: RegionScore): string {
  return `${region.row}-${region.col}`;
}

function extractRegions(feedback: RegionFeedback | null): RegionScore[] {
  if (!feedback) {
    return [];
  }

  const regions =
    feedback.fine_grid?.all_regions ??
    feedback.all_regions ??
    feedback.regions ??
    feedback.region_scores ??
    [];
  return Array.isArray(regions) ? regions : [];
}

function extractRankMap(feedback: RegionFeedback | null): Map<string, number> {
  const ranks = new Map<string, number>();
  if (!feedback) {
    return ranks;
  }

  extractRegions(feedback)
    .filter((region) => typeof region.row === "number" && typeof region.col === "number")
    .slice(0, 3)
    .forEach((region, index) => {
      ranks.set(regionKey(region), index + 1);
    });

  const problemRegions = feedback.fine_grid?.problem_regions ?? feedback.problem_regions ?? [];
  if (Array.isArray(problemRegions)) {
    for (const region of problemRegions) {
      if (!ranks.has(regionKey(region))) {
        ranks.set(regionKey(region), 1);
      }
    }
  }

  return ranks;
}

function regionDisplayValue(region?: RegionScore): string {
  if (region?.insufficient_data) {
    return "n/a";
  }

  const value =
    region?.z_score ??
    region?.normalized_score ??
    region?.score ??
    region?.normalized_error ??
    region?.mean_error ??
    region?.error;
  return typeof value === "number" ? value.toFixed(value > 1 ? 1 : 3) : "";
}

type RegionGridProps = {
  feedback: RegionFeedback | null;
  rows?: number;
  cols?: number;
};

export function RegionGrid({ feedback, rows = 3, cols = 3 }: RegionGridProps) {
  const regions = extractRegions(feedback);
  const rankMap = extractRankMap(feedback);

  return (
    <View style={styles.grid}>
      {Array.from({ length: rows * cols }).map((_, index) => {
        const row = Math.floor(index / cols);
        const col = index % cols;
        const key = `${row}-${col}`;
        const region = regions.find((item) => item.row === row && item.col === col);
        const rank = rankMap.get(key);
        const isPrimary = rank === 1;
        const isSecondary = rank === 2 || rank === 3;

        return (
          <View
            key={key}
            style={[
              styles.cell,
              isPrimary && styles.primaryCell,
              isSecondary && styles.secondaryCell,
            ]}
          >
            {rank ? (
              <View style={[styles.rankBadge, isPrimary ? styles.primaryBadge : styles.secondaryBadge]}>
                <Text style={[styles.rankText, isPrimary ? styles.primaryRankText : styles.secondaryRankText]}>
                  {rank}
                </Text>
              </View>
            ) : null}
            <Text
              style={[
                styles.cellLabel,
                isPrimary && styles.primaryText,
                isSecondary && styles.secondaryText,
              ]}
            >
              {region?.label ?? region?.region ?? CELL_LABELS[index] ?? key}
            </Text>
            <Text
              style={[
                styles.cellValue,
                isPrimary && styles.primaryText,
                isSecondary && styles.secondaryText,
              ]}
            >
              {regionDisplayValue(region)}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  grid: {
    width: "100%",
    aspectRatio: 1,
    borderWidth: 1,
    borderColor: "#263238",
    flexDirection: "row",
    flexWrap: "wrap",
    backgroundColor: "#f8faf7",
  },
  cell: {
    width: "33.3333%",
    height: "33.3333%",
    borderWidth: 0.5,
    borderColor: "#5f6f69",
    alignItems: "center",
    justifyContent: "center",
    padding: 6,
  },
  primaryCell: {
    backgroundColor: "#ffe3dc",
    borderColor: "#b9432e",
  },
  secondaryCell: {
    backgroundColor: "#fff1bf",
    borderColor: "#c58b18",
  },
  rankBadge: {
    alignItems: "center",
    borderRadius: 10,
    height: 20,
    justifyContent: "center",
    position: "absolute",
    right: 5,
    top: 5,
    width: 20,
  },
  primaryBadge: {
    backgroundColor: "#b9432e",
  },
  secondaryBadge: {
    backgroundColor: "#c58b18",
  },
  rankText: {
    fontSize: 11,
    fontWeight: "900",
  },
  primaryRankText: {
    color: "#ffffff",
  },
  secondaryRankText: {
    color: "#ffffff",
  },
  cellLabel: {
    color: "#263238",
    fontSize: 11,
    textAlign: "center",
  },
  cellValue: {
    color: "#52615b",
    fontSize: 12,
    fontWeight: "700",
    marginTop: 4,
  },
  primaryText: {
    color: "#8e2f1e",
  },
  secondaryText: {
    color: "#76500d",
  },
});
