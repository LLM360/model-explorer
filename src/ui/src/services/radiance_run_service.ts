/**
 * @license
 * Copyright 2024 The Model Explorer Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

import {Injectable, signal} from '@angular/core';

import {GraphCollection} from '../components/visualizer/common/input_graph';

export interface RadianceRunSummary {
  runId: string;
  sourceId: string;
  displayName: string;
  sourceKind: string;
  status: string;
  createdAt: string;
  updatedAt: string;
  completedAt?: string | null;
  statusMessage?: string | null;
  graphReady: boolean;
}

interface StartRunResponse {
  run: RadianceRunSummary;
}

interface RunListResponse {
  runs: RadianceRunSummary[];
}

interface GetRunResponse {
  run: RadianceRunSummary;
}

interface RunGraphResponse {
  graphCollections: GraphCollection[];
}

const RUNS_API_PATH = '/api/v1/radiance/runs';

@Injectable({
  providedIn: 'root',
})
export class RadianceRunService {
  readonly loadingRuns = signal<boolean>(false);
  readonly submittingRun = signal<boolean>(false);
  readonly pollingRunId = signal<string | null>(null);
  readonly lastError = signal<string | null>(null);
  readonly runs = signal<RadianceRunSummary[]>([]);

  constructor() {
    void this.refreshRuns();
  }

  async refreshRuns() {
    this.loadingRuns.set(true);
    try {
      const payload = await this.request<RunListResponse>(RUNS_API_PATH);
      this.runs.set(payload.runs);
    } finally {
      this.loadingRuns.set(false);
    }
  }

  async startRun(sourceId: string): Promise<RadianceRunSummary> {
    this.submittingRun.set(true);
    try {
      const payload = await this.request<StartRunResponse>(RUNS_API_PATH, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({sourceId}),
      });
      await this.refreshRuns();
      return payload.run;
    } finally {
      this.submittingRun.set(false);
    }
  }

  async waitForCompletion(runId: string): Promise<RadianceRunSummary> {
    this.pollingRunId.set(runId);
    try {
      while (true) {
        const payload = await this.request<GetRunResponse>(`${RUNS_API_PATH}/${runId}`);
        const run = payload.run;
        this._mergeRun(run);
        if (run.status === 'succeeded' || run.status === 'failed' || run.status === 'cancelled') {
          return run;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 750));
      }
    } finally {
      this.pollingRunId.set(null);
    }
  }

  async loadRunGraph(runId: string): Promise<GraphCollection[]> {
    const payload = await this.request<RunGraphResponse>(`${RUNS_API_PATH}/${runId}/graph`);
    return payload.graphCollections;
  }

  async deleteRun(runId: string): Promise<void> {
    this.submittingRun.set(true);
    try {
      await this.request<{deletedRunId: string}>(`${RUNS_API_PATH}/${runId}`, {
        method: 'DELETE',
      });
      this.runs.set(this.runs().filter((run) => run.runId !== runId));
    } finally {
      this.submittingRun.set(false);
    }
  }

  private _mergeRun(run: RadianceRunSummary) {
    const currentRuns = this.runs();
    const index = currentRuns.findIndex((current) => current.runId === run.runId);
    if (index === -1) {
      this.runs.set([run, ...currentRuns]);
      return;
    }

    const nextRuns = currentRuns.slice();
    nextRuns[index] = run;
    this.runs.set(nextRuns);
  }

  private async request<T>(url: string, init?: RequestInit): Promise<T> {
    this.lastError.set(null);
    const response = await fetch(url, init);
    const payload = (await response.json()) as T & {error?: string};
    if (!response.ok || payload.error) {
      const errorMessage = payload.error || `Request failed with status ${response.status}`;
      this.lastError.set(errorMessage);
      throw new Error(errorMessage);
    }
    return payload;
  }
}
