import { StyleSheet, Text, View } from "react-native";

import type { RegionFeedback, RegionScore } from "../types";

const BAND_LABELS = ["top", "middle", "bottom"];
const UI = {
  ink: "#393D3F",
  paper: "#FDFDFF",
  muted: "#C6C5B9",
  slate: "#546A7B",
  accent: "#62929E",
  softMuted: "#EFEFEB",
  softAccent: "#E7F0F2",
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

function bandLabel(band: string): string {
  return `${band[0].toUpperCase()}${band.slice(1)}`;
}

function bandMessage(band: string, rank?: number): string {
  if (rank === 1) {
    return "Needs attention";
  }
  if (rank === 2 || rank === 3) {
    return "Check this area";
  }
  return "Looks okay";
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
              {region?.label ?? bandLabel(region?.region ?? band)}
            </Text>
            <Text
              style={[
                styles.cellValue,
                isPrimary && styles.primaryText,
                isSecondary && styles.secondaryText,
              ]}
            >
              {bandMessage(band, rank)}
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
    backgroundColor: UI.paper,
    borderRadius: 28,
    elevation: 3,
    gap: 10,
    padding: 14,
    shadowColor: UI.slate,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
  },
  cell: {
    width: "100%",
    minHeight: 76,
    backgroundColor: UI.softMuted,
    borderRadius: 22,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 18,
    paddingVertical: 14,
  },
  primaryCell: {
    backgroundColor: UI.primaryProblemSoft,
  },
  secondaryCell: {
    backgroundColor: UI.secondaryProblemSoft,
  },
  rankBadge: {
    alignItems: "center",
    borderRadius: 14,
    height: 28,
    justifyContent: "center",
    position: "absolute",
    right: 14,
    top: 14,
    width: 28,
  },
  primaryBadge: {
    backgroundColor: UI.primaryProblem,
  },
  secondaryBadge: {
    backgroundColor: UI.secondaryProblem,
  },
  rankText: {
    fontSize: 13,
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
    fontSize: 18,
    fontWeight: "900",
    textAlign: "center",
  },
  cellValue: {
    color: UI.slate,
    fontSize: 13,
    fontWeight: "800",
    marginTop: 6,
  },
  primaryText: {
    color: UI.primaryProblem,
  },
  secondaryText: {
    color: UI.secondaryProblem,
  },
});
