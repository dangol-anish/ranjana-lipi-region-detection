import { StyleSheet, Text, View } from "react-native";

import type { RegionFeedback, RegionScore } from "../types";

const BAND_LABELS = ["top", "middle", "bottom"];
const UI = {
  ink: "#393D3F",
  paper: "#FDFDFF",
  muted: "#C6C5B9",
  slate: "#546A7B",
  softMuted: "#EFEFEB",
  primaryProblem: "#B33B2E",
  primaryProblemSoft: "#F7DFDB",
  secondaryProblem: "#A46A16",
  secondaryProblemSoft: "#F5EBCB",
};

function regionKey(region: RegionScore): string {
  return region.region ?? `${region.row}-${region.col}`;
}

function extractRegions(feedback: RegionFeedback | null): RegionScore[] {
  if (!feedback) {
    return [];
  }

  const regions =
    feedback.broad_bands?.all_regions ??
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

  const problemRegions =
    feedback.broad_bands?.problem_regions ??
    feedback.fine_grid?.problem_regions ??
    feedback.problem_regions ??
    [];
  if (!Array.isArray(problemRegions) || problemRegions.length === 0) {
    return ranks;
  }

  extractRegions(feedback)
    .filter((region) => typeof region.region === "string")
    .slice(0, 3)
    .forEach((region, index) => {
      ranks.set(regionKey(region), index + 1);
    });

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

export function RegionGrid({ feedback }: RegionGridProps) {
  const regions = extractRegions(feedback);
  const rankMap = extractRankMap(feedback);

  return (
    <View style={styles.grid}>
      {BAND_LABELS.map((band) => {
        const region = regions.find((item) => item.region === band);
        const key = band;
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
              {region?.label ?? region?.region ?? band}
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
    height: 240,
    borderWidth: 1,
    borderColor: UI.ink,
    backgroundColor: UI.softMuted,
  },
  cell: {
    width: "100%",
    height: "33.3333%",
    borderWidth: 0.5,
    borderColor: UI.muted,
    alignItems: "center",
    justifyContent: "center",
    padding: 6,
  },
  primaryCell: {
    backgroundColor: UI.primaryProblemSoft,
    borderColor: UI.primaryProblem,
  },
  secondaryCell: {
    backgroundColor: UI.secondaryProblemSoft,
    borderColor: UI.secondaryProblem,
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
    backgroundColor: UI.primaryProblem,
  },
  secondaryBadge: {
    backgroundColor: UI.secondaryProblem,
  },
  rankText: {
    fontSize: 11,
    fontWeight: "900",
  },
  primaryRankText: {
    color: UI.paper,
  },
  secondaryRankText: {
    color: UI.paper,
  },
  cellLabel: {
    color: UI.ink,
    fontSize: 11,
    textAlign: "center",
  },
  cellValue: {
    color: UI.slate,
    fontSize: 12,
    fontWeight: "700",
    marginTop: 4,
  },
  primaryText: {
    color: UI.primaryProblem,
  },
  secondaryText: {
    color: UI.secondaryProblem,
  },
});
