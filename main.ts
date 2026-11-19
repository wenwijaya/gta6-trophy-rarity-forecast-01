// gta6-trophy-rarity-forecast-01 — forecast each GTA VI trophy's Exophase earn
// rate at a case's target, as five quantiles, from the case's own history.
//
// Model (per trophy, all in logit space, rates clamped to 0.01..99.99):
//   1. Trend: weighted least squares of logit(rate) on u = ln(1 + hours since
//      the 2026-11-19T00:00Z launch), fitted on the trophy's change points in
//      the last 36 hours only, recency-weighted (12 h half-life). Only the
//      recent window is used because launch-night growth is explosive and does
//      not persist: a full-history fit extrapolates the early surge and
//      overshoots a saturating curve by ~1 logit (measured on synthetic
//      final-like cases). The slope is shrunk by its own t-statistic, so a
//      noisy or short history stays near persistence, and the extrapolated
//      move is capped at 1.0 logit.
//   2. Refresh prior: Exophase refreshes its page on its own schedule, and a
//      refresh dilutes every rate (newly tracked players hold few trophies).
//      When a case's whole history is one unchanged snapshot — the page has
//      not moved at all, so the trend fit can say nothing about dilution —
//      and the horizon is short (<= 1 day), the median is shifted down by
//      delta = min(2.0, 0.33 * stale-hours-to-target) logit, scaled by
//      sqrt(1 - r/100) so a saturated trophy (100%) does not move. The 0.33
//      per hour is calibrated on the kit's own record: the launch-night page
//      sat unchanged for ~4.7 h, then refreshed with players-tracked 638 ->
//      3868, and the house persistence baseline's practice loss of ~1.51
//      implies a ~1.6-logit fall, i.e. rates near a fifth of their last read.
//      When the history already contains refreshes, no prior is applied: the
//      trend fit has seen dilution happen and already accounts for it.
//   3. Spread: a random-walk term (0.5 logit per sqrt-day, as in the house
//      baseline) combined in quadrature with the fit's residual volatility
//      scaled by horizon/history-span, the slope's standard error times the
//      extrapolation distance in u, and an allowance for refresh jumps; the
//      total is scaled by 0.8 (a full-width spread loses ~0.05-0.07 of pinball
//      on synthetic finals and the narrower band still wins under a simulated
//      -0.8-logit refresh shock), floored at 0.45 logit (the baseline's
//      90-minute band was far too narrow — it covered ~0% of a refresh) and
//      capped at 2.0.
//
// Node 22 standard library only, offline, deterministic: every number comes
// from the case file, never from the wall clock or any randomness.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Launch instant: u = ln(1 + hours since this instant). */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** Recency half-life for the trend fit, in hours. */
const HALF_LIFE_H = 12;
/** Only reads from the last WINDOW_H hours before the last read enter the fit. */
const WINDOW_H = 36;
/** Random-walk spread, logit per square-root day (house baseline's value). */
const SIGMA_RW = 0.5;
/** Cap on the extrapolated median move from the last read, in logit units. */
const MOVE_CAP = 1.0;
/** Multiplier on the combined spread (tuned on synthetic final-like cases). */
const SPREAD_MULT = 0.8;

interface TrophyDef { id: number; tier: string; points: number; secret: boolean }
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: TrophyDef[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct);
  return Math.log(p / (100 - p));
};
/** Numerically stable 100 * sigmoid(x). */
const invLogit = (x: number): number => {
  if (x >= 0) return 100 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return (100 * e) / (1 + e);
};
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const hoursSinceLaunch = (ms: number): number => Math.max(0, (ms - LAUNCH_MS) / 3_600_000);
const uOf = (ms: number): number => Math.log1p(hoursSinceLaunch(ms));

interface Fit {
  /** Intercept, in logit units. */
  a: number;
  /** Slope per unit of u. */
  b: number;
  /** Standard error of the slope (Infinity when unidentified). */
  seB: number;
  /** Weighted residual standard deviation, in logit units. */
  resid: number;
  /** Weighted mean of u. */
  uMean: number;
  /** Number of points used. */
  n: number;
}

/**
 * Weighted least squares of y on u (with intercept). Returns a zero slope
 * with infinite standard error when fewer than 3 distinct u values are seen.
 */
function wls(points: Array<{ u: number; y: number; w: number }>): Fit {
  const distinctU = new Set(points.map((p) => p.u)).size;
  const sw = points.reduce((s, p) => s + p.w, 0);
  const uMean = points.reduce((s, p) => s + p.w * p.u, 0) / sw;
  const yMean = points.reduce((s, p) => s + p.w * p.y, 0) / sw;
  if (points.length < 3 || distinctU < 3 || sw <= 0) {
    return { a: yMean, b: 0, seB: Infinity, resid: 0, uMean, n: points.length };
  }
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    sxx += p.w * (p.u - uMean) * (p.u - uMean);
    sxy += p.w * (p.u - uMean) * (p.y - yMean);
  }
  if (sxx <= 1e-12) return { a: yMean, b: 0, seB: Infinity, resid: 0, uMean, n: points.length };
  const b = sxy / sxx;
  const a = yMean - b * uMean;
  let sse = 0;
  for (const p of points) {
    const r = p.y - (a + b * p.u);
    sse += p.w * r * r;
  }
  // Effective sample size, so a run of near-identical weights cannot fake
  // precision; dof at least 1.
  const w2 = points.reduce((s, p) => s + p.w * p.w, 0);
  const nEff = (sw * sw) / Math.max(w2, 1e-12);
  const dof = Math.max(1, nEff - 2);
  const resid = Math.sqrt(Math.max(0, sse / Math.max(sw, 1e-12)));
  // Weighted residual variance (dof-corrected via the effective sample
  // size) over Sxx gives the slope's variance.
  const varB = (sse / Math.max(sw, 1e-12)) * (nEff / Math.max(dof, 1)) / sxx;
  return { a, b, seB: Math.sqrt(Math.max(0, varB)), resid, uMean, n: points.length };
}

/**
 * Forecast one trophy. `reads` are the case's history in order; `idx` the
 * trophy's position in each read's rates; `ctx` carries case-level facts
 * (whether the whole history is one unchanged snapshot, whether any refresh
 * jump was seen, horizon and target in u).
 */
function forecastTrophy(
  reads: Read[],
  idx: number,
  ctx: { flat: boolean; refreshJumps: number; horizonMin: number; uTarget: number; targetMs: number },
): number[] {
  // Last non-null rate; fall back to 1% with a wide spread when a trophy was
  // never read (a valid file must still be written for it).
  let lastRate: number | null = null;
  for (let j = reads.length - 1; j >= 0; j--) {
    const r = reads[j].rates[idx];
    if (r !== null && r !== undefined && Number.isFinite(r)) { lastRate = r; break; }
  }
  const yLast = lastRate === null ? logit(1) : logit(lastRate);
  if (reads.length === 0 || lastRate === null) {
    return Z.map((z) => round4(clampPct(invLogit(yLast + z * 2.0))));
  }
  const tLast = Date.parse(reads[reads.length - 1].at);
  const tFirst = Date.parse(reads[0].at);
  const spanDays = Math.max((tLast - tFirst) / 86_400_000, 0.01);
  const hDays = ctx.horizonMin / 1440;

  // --- Trend fit on change points in the recent window only. ---
  const points: Array<{ u: number; y: number; w: number }> = [];
  let prevRate: number | null = null;
  for (let j = 0; j < reads.length; j++) {
    const r = reads[j].rates[idx];
    if (r === null || r === undefined || !Number.isFinite(r)) continue;
    const t = Date.parse(reads[j].at);
    if (tLast - t > WINDOW_H * 3_600_000) { prevRate = r; continue; }
    const isLast = j === reads.length - 1;
    if (r !== prevRate || isLast) {
      const w = Math.pow(0.5, Math.max(0, tLast - t) / (HALF_LIFE_H * 3_600_000));
      points.push({ u: uOf(t), y: logit(r), w });
    }
    prevRate = r;
  }
  const fit = wls(points);
  // Shrink the slope by its t-statistic: t^2 / (t^2 + 3) is ~0 for noise,
  // ~1 for a slope the data really supports.
  let shrink = 0;
  if (Number.isFinite(fit.seB) && fit.seB > 0 && fit.b !== 0) {
    const t = Math.abs(fit.b) / fit.seB;
    shrink = (t * t) / (t * t + 3);
  }
  const bShrunk = fit.b * shrink;
  let m = fit.a + bShrunk * ctx.uTarget;
  // Anchor the extrapolation at the last read rather than at the weighted
  // mean when the fit is degenerate, and cap the total move.
  if (!Number.isFinite(m)) m = yLast;
  m = yLast + Math.max(-MOVE_CAP, Math.min(MOVE_CAP, m - yLast));

  // --- Blend toward persistence by fit quality, then the refresh prior. ---
  const wTrend = 0.35 + 0.45 * shrink;
  m = wTrend * m + (1 - wTrend) * yLast;
  m = yLast + Math.max(-MOVE_CAP, Math.min(MOVE_CAP, m - yLast));
  let refreshPrior = false;
  if (ctx.flat && ctx.horizonMin <= 1440) {
    // Stale hours from the first (unchanged) read to the target.
    const staleH = (ctx.targetMs - tFirst) / 3_600_000;
    const delta = Math.min(2.0, 0.33 * Math.max(0, staleH));
    const scale = Math.sqrt(Math.max(0, 1 - (lastRate as number) / 100));
    m += -delta * scale;
    m = yLast + Math.max(-3.2, Math.min(3.2, m - yLast));
    refreshPrior = true;
  }
  if (!Number.isFinite(m)) m = yLast;

  // --- Spread: random walk + residual volatility + slope uncertainty +
  // refresh-jump allowance, in quadrature; floored and capped. ---
  const base = SIGMA_RW * Math.sqrt(hDays);
  const residTerm = fit.resid * Math.sqrt(hDays / Math.max(spanDays, 0.25));
  const slopeUnc = Number.isFinite(fit.seB)
    ? fit.seB * Math.abs(ctx.uTarget - fit.uMean)
    : 0.5 * Math.sqrt(hDays);
  const refreshAllow = refreshPrior ? 0.3 : ctx.refreshJumps > 0 ? 0.35 : 0.15;
  let s = Math.sqrt(
    base * base + residTerm * residTerm + slopeUnc * slopeUnc + refreshAllow * refreshAllow,
  );
  s = Math.max(0.45, Math.min(2.0, s * SPREAD_MULT));
  if (!Number.isFinite(s)) s = 0.5;

  return Z.map((z) => round4(clampPct(invLogit(m + z * s))));
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const reads = Array.isArray(c.history) ? c.history : [];
  // Case-level facts: is the whole history one unchanged snapshot, and did
  // any refresh jump (a >2% move in players-tracked) happen?
  let flat = reads.length > 0;
  let refreshJumps = 0;
  for (let j = 1; j < reads.length; j++) {
    const a = reads[j - 1];
    const b = reads[j];
    if (a.players_tracked !== b.players_tracked) flat = false;
    if (
      a.players_tracked > 0 &&
      Math.abs(b.players_tracked - a.players_tracked) / a.players_tracked > 0.02
    ) {
      refreshJumps++;
    }
    for (let i = 0; i < Math.min(a.rates.length, b.rates.length); i++) {
      if (a.rates[i] !== b.rates[i]) flat = false;
    }
  }
  if (reads.length > 0 && reads[0].rates.length !== c.trophies.length) flat = false;
  const uTarget = uOf(Date.parse(c.target));
  const ctx = { flat, refreshJumps, horizonMin: c.horizon_minutes, uTarget, targetMs: Date.parse(c.target) };
  const trophies = c.trophies.map((t, i) => ({ id: t.id, q: forecastTrophy(reads, i, ctx) }));
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
