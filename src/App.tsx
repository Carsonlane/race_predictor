import React, { useMemo, useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { Plus, Trash2, Save, UserPlus, Calculator, LogIn } from "lucide-react";

/**
 * Race Predictor — Single-file React app
 * - Tailwind + shadcn/ui components
 * - LocalStorage persistence (demo accounts / profiles)
 * - Heuristic performance model blending PR-derived Riegel predictions
 *   with workout-derived fitness signals (MAS/ASR/Threshold proxies)
 *
 * Notes on the model (concise):
 *   1) Base curve from latest/best PR using Riegel: t2 = t1 * (d2/d1)^b, with b dependent on athlete type.
 *   2) Workouts mapped to fitness proxies:
 *      - MAS (max aerobic speed) proxy from short reps (150–600m): adjust by work:rest.
 *      - Threshold proxy from longer reps (800–3000m): pace stabilized by recovery factor.
 *      - ASR (anaerobic speed reserve) proxy from near-400m pace vs MAS.
 *   3) Recency weighting: exponential decay (half-life 21 days) to emphasize recent sessions.
 *   4) Athlete-type weighting mixes short vs long signals.
 *   5) Final prediction = geometric blend of PR curve and workout curve per event.
 */

// ---------------------------- Utilities ----------------------------
const DAY = 24 * 3600 * 1000;
const toISODate = (d) => new Date(d).toISOString().slice(0,10);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

function parseTimeToSeconds(str) {
  // Accepts formats like "1:45.5", "3:59", "59.3", "14:30", "00:50"
  if (!str) return NaN;
  const parts = String(str).trim().split(":");
  if (parts.length === 1) return parseFloat(parts[0]);
  if (parts.length === 2) {
    const [m, s] = parts;
    return (parseInt(m, 10) || 0) * 60 + (parseFloat(s) || 0);
  }
  if (parts.length === 3) {
    const [h, m, s] = parts;
    return (parseInt(h,10)||0)*3600 + (parseInt(m,10)||0)*60 + (parseFloat(s)||0);
  }
  return NaN;
}

function formatSeconds(t) {
  if (!isFinite(t)) return "—";
  if (t >= 3600) {
    const h = Math.floor(t/3600);
    const m = Math.floor((t%3600)/60);
    const s = (t%60).toFixed(1).padStart(4, "0");
    return `${h}:${String(m).padStart(2,"0")}:${String(s).padStart(4,"0")}`;
  }
  const m = Math.floor(t/60);
  const s = (t%60).toFixed(1).padStart(4,"0");
  return `${m}:${String(s).padStart(4,"0")}`;
}

// Meters for common events
const EVENTS = [
  { key: "400", m: 400 },
  { key: "800", m: 800 },
  { key: "1500", m: 1500 },
  { key: "mile", m: 1609 },
  { key: "3000", m: 3000 },
  { key: "5000", m: 5000 },
  { key: "10000", m: 10000 },
  { key: "half", m: 21097.5 },
  { key: "marathon", m: 42195 },
];

const ATHLETE_TYPES = [
  "400/800","800","800/1500/miler","1500/miler","1500/5000","5000","5000/10000","10000/marathon","marathon"
];

// Default Riegel exponent by athlete type (lower = better endurance scaling)
const RIEGEL_BY_TYPE = {
  "400/800": 1.12,
  "800": 1.10,
  "800/1500/miler": 1.085,
  "1500/miler": 1.08,
  "1500/5000": 1.07,
  "5000": 1.065,
  "5000/10000": 1.06,
  "10000/marathon": 1.055,
  "marathon": 1.05,
};

// Workout weighting by type: [shortRepWeight, longRepWeight]
const TYPE_WEIGHTS = {
  "400/800": [0.70, 0.30],
  "800": [0.65, 0.35],
  "800/1500/miler": [0.58, 0.42],
  "1500/miler": [0.52, 0.48],
  "1500/5000": [0.45, 0.55],
  "5000": [0.38, 0.62],
  "5000/10000": [0.32, 0.68],
  "10000/marathon": [0.25, 0.75],
  "marathon": [0.18, 0.82],
};

// ---------------------------- Storage ----------------------------
const LS_KEY = "race-predictor-profiles-v1";
const loadProfiles = () => {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || "[]"); } catch { return []; }
};
const saveProfiles = (arr) => localStorage.setItem(LS_KEY, JSON.stringify(arr));

// ---------------------------- Model Core ----------------------------
function riegelPredict(t1, d1, d2, b) { return t1 * Math.pow(d2/d1, b); }

function decayWeight(dateStr, halfLifeDays = 21) {
  const now = Date.now();
  const t = new Date(dateStr).getTime();
  if (isNaN(t)) return 0.0;
  const k = Math.pow(0.5, (now - t) / (halfLifeDays * DAY));
  return clamp(k, 0, 1);
}

/**
 * Extract proxies from a single workout
 * workout = { reps, repDistanceM, repTimeStr, restSeconds, date }
 */
function proxiesFromWorkout(w) {
  const repSec = parseTimeToSeconds(w.repTimeStr);
  if (!isFinite(repSec) || repSec <= 0) return null;
  const vRep = w.repDistanceM / repSec; // m/s
  const work = repSec, rest = w.restSeconds || 0;
  const wr = work > 0 ? rest / work : 0;

  // Recovery attenuation: more rest => closer to true rep speed; less rest => more aerobic load
  const recAtten = 1 / (1 + 0.6 / (wr + 0.05)); // smoothly maps wr → (0,1]

  // Short reps drive MAS when distance <= 600m
  const isShort = w.repDistanceM <= 600;
  const isLong = w.repDistanceM >= 800;

  // MAS proxy: approach rep speed but attenuate toward realistic MAS
  const masProxy = isShort ? vRep * (0.86 + 0.12 * recAtten) : null; // ~ 3-5% below 3k pace speed as guard

  // Threshold proxy: long reps with limited rest approximate T-pace
  const tProxy = isLong ? vRep * (0.74 + 0.20 * (1 - Math.tanh(wr))) : null; // less rest => more threshold-like

  // 400 speed proxy (for ASR): prefer distances 300–500m
  const asrProxy = (w.repDistanceM >= 300 && w.repDistanceM <= 500) ? vRep : null;

  return { masProxy, tProxy, asrProxy };
}

function aggregateFitness(workouts) {
  let masSum=0, masW=0, tSum=0, tW=0, asrSum=0, asrW=0;
  for (const w of workouts) {
    const p = proxiesFromWorkout(w);
    if (!p) continue;
    const wt = decayWeight(w.date);
    if (p.masProxy) { masSum += p.masProxy * wt; masW += wt; }
    if (p.tProxy)   { tSum   += p.tProxy   * wt; tW   += wt; }
    if (p.asrProxy) { asrSum += p.asrProxy * wt; asrW += wt; }
  }
  const mas = masW>0 ? masSum/masW : NaN;   // m/s
  const t   = tW>0 ? tSum/tW       : NaN;   // m/s
  const asr = asrW>0 ? asrSum/asrW : NaN;   // m/s @ ~400 rep
  return { mas, t, asr };
}

function workoutTimeForDistance(meters, fit, typeKey) {
  // Convert fitness proxies into an expected time at a given distance.
  // Construct a synthetic curve using:
  //   - Near 1500m anchor from MAS (~vVO2max): t1500 ≈ 1500 / (0.92 * MAS)
  //   - Threshold anchor at 10k-ish: t10k ≈ 10000 / (T * 0.98)
  //   - Short-distance correction from ASR (benefits sub-800)
  // Then derive exponent b_w from anchors and interpolate.
  const [wShort, wLong] = TYPE_WEIGHTS[typeKey] || [0.5, 0.5];
  const mas = fit.mas, T = fit.t, asr = fit.asr;

  // Guard rails with typical values if missing
  const masUse = isFinite(mas) ? mas : 5.5;     // m/s (~3:02/km)
  const tUse   = isFinite(T)   ? T   : 4.2;     // m/s (~3:58/km)
  const asrUse = isFinite(asr) ? asr : masUse;  // fallback

  const t1500 = 1500 / (0.92 * masUse);
  const t10k  = 10000 / (0.98 * tUse);

  // Compute exponent from two anchors: t ~ d^b
  const bW = Math.log(t10k / t1500) / Math.log(10000/1500);
  // Blend with type emphasis (longer types push toward endurance)
  const bAdj = clamp(bW * (0.65 + 0.35*wLong), 1.03, 1.16);

  // Scale constant from 1500 anchor: t = a * d^{bAdj}
  const a = t1500 / Math.pow(1500, bAdj);
  let tPred = a * Math.pow(meters, bAdj);

  // Short distance correction using ASR for <= 800m
  if (meters <= 800) {
    const pure = meters / (0.90 * asrUse); // optimistic raw from ASR
    // Blend with curve: more short-weight bias for speed types
    const bias = 0.35 + 0.45 * wShort; // 0.35–0.80
    tPred = Math.exp(bias * Math.log(pure) + (1-bias) * Math.log(tPred));
  }

  return tPred;
}

function blendPredictions(prTime, prDist, meters, typeKey, workouts) {
  const b = RIEGEL_BY_TYPE[typeKey] || 1.08;
  const base = riegelPredict(prTime, prDist, meters, b);
  const fit = aggregateFitness(workouts);
  const wkt = workoutTimeForDistance(meters, fit, typeKey);

  // Confidence weighting: depends on number, recency of workouts
  const n = workouts.length;
  const recentWt = workouts.reduce((s,w)=>s+decayWeight(w.date),0);
  const conf = clamp( (recentWt / Math.max(1,n)) * 0.9, 0, 0.9); // 0..0.9
  const dataWt = clamp(0.25 + 0.6*conf, 0.25, 0.85); // ensure some floor & cap

  // Geometric blend to avoid dominance: exp( w*ln(wkt) + (1-w)*ln(base) )
  const ln = (x)=>Math.log(Math.max(1e-6,x));
  const tBlend = Math.exp(dataWt*ln(wkt) + (1-dataWt)*ln(base));

  return { base, wkt, tBlend, fitness: fit, dataWt };
}

// ---------------------------- UI Components ----------------------------
function Section({title, children, subtitle}){
  return (
    <div className="space-y-2">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
      <div className="mt-2">{children}</div>
    </div>
  );
}

function Row({children}){ return <div className="grid grid-cols-1 md:grid-cols-3 gap-3">{children}</div>; }

function SmallField({label, children}){
  return (
    <div className="space-y-1">
      <Label className="text-sm">{label}</Label>
      {children}
    </div>
  );
}

function PredictionsTable({ preds }){
  return (
    <div className="overflow-x-auto rounded-2xl border">
      <table className="w-full text-sm">
        <thead className="bg-muted/40">
          <tr>
            <th className="text-left p-3">Event</th>
            <th className="text-right p-3">From PR</th>
            <th className="text-right p-3">From Workouts</th>
            <th className="text-right p-3">Blended</th>
          </tr>
        </thead>
        <tbody>
          {preds.map((p)=> (
            <tr key={p.event} className="border-t">
              <td className="p-3 font-medium">{p.event}</td>
              <td className="p-3 text-right tabular-nums">{formatSeconds(p.base)}</td>
              <td className="p-3 text-right tabular-nums">{formatSeconds(p.wkt)}</td>
              <td className="p-3 text-right tabular-nums font-semibold">{formatSeconds(p.tBlend)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------- App ----------------------------
export default function App(){
  const [athleteType, setAthleteType] = useState("1500/miler");
  const [sex, setSex] = useState("male");
  const [age, setAge] = useState(20);
  const [prEvent, setPrEvent] = useState("1500");
  const [prTime, setPrTime] = useState("4:00.0");
  const [prDate, setPrDate] = useState(toISODate(new Date()));
  const [workouts, setWorkouts] = useState([]);
  const [showAcct, setShowAcct] = useState(false);
  const [profiles, setProfiles] = useState(loadProfiles());
  const [profileName, setProfileName] = useState("");
  const [useDemo, setUseDemo] = useState(true);

  // Demo seed (toggleable)
  useEffect(()=>{
    if (useDemo && workouts.length===0) {
      setWorkouts([
        { id: crypto.randomUUID(), reps: 6, repDistanceM: 400, repTimeStr: "62.0", restSeconds: 60, date: toISODate(Date.now()-7*DAY)},
        { id: crypto.randomUUID(), reps: 4, repDistanceM: 300, repTimeStr: "45.0", restSeconds: 90, date: toISODate(Date.now()-10*DAY)},
        { id: crypto.randomUUID(), reps: 5, repDistanceM: 1000, repTimeStr: "2:50", restSeconds: 90, date: toISODate(Date.now()-21*DAY)},
        { id: crypto.randomUUID(), reps: 3, repDistanceM: 2000, repTimeStr: "5:54", restSeconds: 120, date: toISODate(Date.now()-30*DAY)},
      ]);
    }
  },[useDemo]);

  const prMeters = useMemo(()=>{
    const ev = EVENTS.find(e=>e.key===prEvent);
    return ev? ev.m : 1500;
  },[prEvent]);

  const predictions = useMemo(()=>{
    const tPR = parseTimeToSeconds(prTime);
    if (!isFinite(tPR)) return [];
    return EVENTS.map(ev => {
      const {base, wkt, tBlend} = blendPredictions(tPR, prMeters, ev.m, athleteType, workouts);
      return { event: ev.key, base, wkt, tBlend };
    });
  },[athleteType, prEvent, prTime, prMeters, workouts]);

  function addWorkout(){
    setWorkouts(w=>[
      ...w,
      { id: crypto.randomUUID(), reps: 4, repDistanceM: 400, repTimeStr: "65.0", restSeconds: 60, date: toISODate(new Date()) }
    ]);
  }
  function updateWorkout(id, patch){ setWorkouts(ws=>ws.map(w=>w.id===id? {...w, ...patch}: w)); }
  function delWorkout(id){ setWorkouts(ws=>ws.filter(w=>w.id!==id)); }

  function saveProfile(){
    if (!profileName) return;
    const prof = {
      id: crypto.randomUUID(),
      name: profileName,
      created: toISODate(new Date()),
      athlete: { type: athleteType, sex, age, pr: { event: prEvent, time: prTime, date: prDate } },
      workouts,
      predictions,
    };
    const next = [prof, ...profiles];
    setProfiles(next); saveProfiles(next);
    setShowAcct(false); setProfileName("");
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-black text-slate-100 p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-2xl bg-cyan-400/20 ring-1 ring-cyan-400/40 flex items-center justify-center text-cyan-300 font-bold">RL</div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold tracking-tight">Race Predictor</h1>
              <p className="text-xs md:text-sm text-cyan-200/80">Blending PR curves with workout‑driven physiology (MAS/ASR/T)</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="secondary" className="rounded-2xl" onClick={()=>setShowAcct(true)}>
              <UserPlus className="h-4 w-4 mr-2"/> Save Profile
            </Button>
          </div>
        </div>

        <Card className="bg-white/5 backdrop-blur border-white/10 rounded-3xl">
          <CardContent className="p-4 md:p-6 space-y-6">
            {/* Athlete */}
            <Section title="Athlete Profile" subtitle="Enter baseline information and a reference PR.">
              <Row>
                <SmallField label="Athlete Type">
                  <Select value={athleteType} onValueChange={setAthleteType}>
                    <SelectTrigger className="rounded-xl"><SelectValue/></SelectTrigger>
                    <SelectContent>
                      {ATHLETE_TYPES.map(t=> <SelectItem key={t} value={t}>{t}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </SmallField>
                <SmallField label="Sex">
                  <Select value={sex} onValueChange={setSex}>
                    <SelectTrigger className="rounded-xl"><SelectValue/></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="male">Male</SelectItem>
                      <SelectItem value="female">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </SmallField>
                <SmallField label="Age">
                  <Input type="number" className="rounded-xl" value={age} onChange={e=>setAge(parseInt(e.target.value||"0",10))}/>
                </SmallField>
              </Row>
              <Row>
                <SmallField label="PR Event">
                  <Select value={prEvent} onValueChange={setPrEvent}>
                    <SelectTrigger className="rounded-xl"><SelectValue/></SelectTrigger>
                    <SelectContent>
                      {EVENTS.map(e=> <SelectItem key={e.key} value={e.key}>{e.key}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </SmallField>
                <SmallField label="PR Time (m:s.ms)">
                  <Input placeholder="e.g., 3:59.9" className="rounded-xl" value={prTime} onChange={e=>setPrTime(e.target.value)}/>
                </SmallField>
                <SmallField label="PR Date">
                  <Input type="date" className="rounded-xl" value={prDate} onChange={e=>setPrDate(e.target.value)}/>
                </SmallField>
              </Row>
            </Section>

            <Separator className="bg-white/10"/>

            {/* Workouts */}
            <Section title="Key Workouts" subtitle="Add representative interval sessions. Recent, specific work will influence predictions more.">
              <div className="space-y-3">
                {workouts.map(w=> (
                  <div key={w.id} className="grid grid-cols-1 md:grid-cols-6 gap-2 items-end bg-white/5 rounded-2xl p-3 border border-white/10">
                    <SmallField label="# Reps">
                      <Input type="number" className="rounded-xl" value={w.reps} onChange={e=>updateWorkout(w.id,{reps:parseInt(e.target.value||"0",10)})}/>
                    </SmallField>
                    <SmallField label="Rep Distance (m)">
                      <Input type="number" className="rounded-xl" value={w.repDistanceM} onChange={e=>updateWorkout(w.id,{repDistanceM:parseInt(e.target.value||"0",10)})}/>
                    </SmallField>
                    <SmallField label="Rep Time (m:s.ms)">
                      <Input className="rounded-xl" value={w.repTimeStr} onChange={e=>updateWorkout(w.id,{repTimeStr:e.target.value})}/>
                    </SmallField>
                    <SmallField label="Rest (sec)">
                      <Input type="number" className="rounded-xl" value={w.restSeconds} onChange={e=>updateWorkout(w.id,{restSeconds:parseInt(e.target.value||"0",10)})}/>
                    </SmallField>
                    <SmallField label="Date">
                      <Input type="date" className="rounded-xl" value={w.date} onChange={e=>updateWorkout(w.id,{date:e.target.value})}/>
                    </SmallField>
                    <div className="flex gap-2 md:justify-end">
                      <Button variant="destructive" className="rounded-xl" onClick={()=>delWorkout(w.id)}>
                        <Trash2 className="h-4 w-4"/>
                      </Button>
                    </div>
                  </div>
                ))}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Switch checked={useDemo} onCheckedChange={setUseDemo}/>
                    <span className="text-sm text-muted-foreground">Seed with demo workouts</span>
                  </div>
                  <Button className="rounded-2xl" onClick={addWorkout}><Plus className="h-4 w-4 mr-2"/>Add Workout</Button>
                </div>
              </div>
            </Section>

            <Separator className="bg-white/10"/>

            {/* Predictions */}
            <Section title="Predicted Marks" subtitle="Blended estimates across events. Hover over cells (title) for details.">
              <PredictionsTable preds={predictions} />
              <div className="text-xs text-muted-foreground leading-relaxed mt-2">
                <p>
                  Model blend weight toward workouts adapts to recency/volume. Short‑rep influence vs long‑rep influence
                  varies by athlete type. Times are expressed as minutes:seconds.tenths. This is a heuristic tool, not a guarantee.
                </p>
              </div>
            </Section>
          </CardContent>
        </Card>

        {/* Saved Profiles */}
        <Card className="bg-white/5 backdrop-blur border-white/10 rounded-3xl">
          <CardContent className="p-4 md:p-6 space-y-4">
            <Section title="Saved Profiles" subtitle="Local (device) storage for demo accounts.">
              {profiles.length===0 ? (
                <p className="text-sm text-muted-foreground">No profiles saved yet.</p>
              ) : (
                <div className="grid md:grid-cols-2 gap-3">
                  {profiles.map(p=> (
                    <div key={p.id} className="rounded-2xl border border-white/10 p-3 bg-white/5">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-semibold">{p.name}</div>
                          <div className="text-xs text-muted-foreground">{p.athlete.type} · {p.athlete.sex} · age {p.athlete.age}</div>
                        </div>
                        <Button size="sm" variant="destructive" className="rounded-xl" onClick={()=>{const next=profiles.filter(x=>x.id!==p.id); setProfiles(next); saveProfiles(next);}}>Delete</Button>
                      </div>
                      <div className="mt-2 text-xs">PR: {p.athlete.pr.event} — {p.athlete.pr.time} ({p.athlete.pr.date})</div>
                      <div className="mt-2 text-xs">Predicted mile: <span className="font-mono">{formatSeconds((p.predictions.find(x=>x.event==='mile')||{}).tBlend)}</span></div>
                    </div>
                  ))}
                </div>
              )}
            </Section>
          </CardContent>
        </Card>
      </div>

      {/* Save Profile Dialog */}
      <Dialog open={showAcct} onOpenChange={setShowAcct}>
        <DialogContent className="rounded-3xl">
          <DialogHeader>
            <DialogTitle>Create Account (Local Demo)</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <Label>Display Name</Label>
            <Input placeholder="e.g., Carson — Fall Build" value={profileName} onChange={e=>setProfileName(e.target.value)} className="rounded-xl"/>
            <p className="text-xs text-muted-foreground">This demo stores data only in your browser. For multi-device sync, wire this UI to a backend (Flask/FastAPI + SQLite/Postgres).</p>
          </div>
          <DialogFooter>
            <Button onClick={saveProfile} className="rounded-2xl"><Save className="h-4 w-4 mr-2"/>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
