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

import {CommonModule} from '@angular/common';
import {Component, computed, inject, Inject, signal} from '@angular/core';
import {FormControl, ReactiveFormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {ModelLoaderServiceInterface} from '../../common/model_loader_service_interface';

import {
  MountedBrowserEntry,
  RadianceModelSource,
  RadianceSourceService,
} from '../../services/radiance_source_service';
import {RadianceRunService} from '../../services/radiance_run_service';

@Component({
  selector: 'radiance-source-input',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatCheckboxModule,
    MatIconModule,
    MatProgressSpinnerModule,
    ReactiveFormsModule,
  ],
  templateUrl: './radiance_source_input.ng.html',
  styleUrls: ['./radiance_source_input.scss'],
})
export class RadianceSourceInput {
  private readonly radianceSourceService = inject(RadianceSourceService);
  private readonly radianceRunService = inject(RadianceRunService);

  readonly mountedDisplayName = new FormControl<string>('');
  readonly huggingFaceRepoId = new FormControl<string>('');
  readonly huggingFaceRevision = new FormControl<string>('main');
  readonly huggingFaceDisplayName = new FormControl<string>('');
  readonly huggingFaceAccessToken = new FormControl<string>('');
  readonly trustRemoteCode = new FormControl<boolean>(false, {nonNullable: true});

  readonly sources = this.radianceSourceService.sources;
  readonly mountedBrowser = this.radianceSourceService.mountedBrowser;
  readonly loadingSources = this.radianceSourceService.loadingSources;
  readonly loadingMountedEntries = this.radianceSourceService.loadingMountedEntries;
  readonly submitting = this.radianceSourceService.submitting;
  readonly lastError = this.radianceSourceService.lastError;
  readonly runError = this.radianceRunService.lastError;
  readonly runs = this.radianceRunService.runs;
  readonly loadingRuns = this.radianceRunService.loadingRuns;
  readonly submittingRun = this.radianceRunService.submittingRun;
  readonly pollingRunId = this.radianceRunService.pollingRunId;
  readonly statusMessage = signal<string | null>(null);
  readonly selectedSourceId = signal<string | null>(null);
  readonly hasUploadedModels = computed(() =>
    this.sources().some((source) => source.kind === 'staged_upload'),
  );

  constructor(
    @Inject('ModelLoaderService')
    private readonly modelLoaderService: ModelLoaderServiceInterface,
  ) {}

  async browseMounted(subpath = '') {
    await this.radianceSourceService.browseMounted(subpath);
  }

  async selectCurrentMountedPath() {
    const currentSubpath = this.mountedBrowser().currentSubpath;
    if (!currentSubpath) {
      return;
    }
    await this.registerMountedSource(currentSubpath);
  }

  async handleMountedEntryClick(entry: MountedBrowserEntry) {
    if (entry.entryType === 'directory') {
      await this.browseMounted(entry.subpath);
      return;
    }
    await this.registerMountedSource(entry.subpath);
  }

  async addFiles(files: File[]) {
    for (const file of files) {
      await this.radianceSourceService.uploadModel(file);
      this.statusMessage.set(`Staged upload ${file.name}`);
    }
  }

  async handleUpload(input: HTMLInputElement) {
    const files = Array.from(input.files || []);
    await this.addFiles(files);
    input.value = '';
  }

  async submitHuggingFaceSource() {
    const repoId = this.huggingFaceRepoId.value?.trim() || '';
    if (!repoId) {
      return;
    }

    const response = await this.radianceSourceService.addHuggingFaceSource({
      repoId,
      revision: this.huggingFaceRevision.value?.trim() || '',
      displayName: this.huggingFaceDisplayName.value?.trim() || '',
      accessToken: this.huggingFaceAccessToken.value?.trim() || '',
      trustRemoteCode: this.trustRemoteCode.value,
    });
    this.statusMessage.set(
      response.authAccepted
        ? `Registered Hugging Face source ${response.modelSource.display_name} with transient auth.`
        : `Registered Hugging Face source ${response.modelSource.display_name}.`,
    );
    this.huggingFaceAccessToken.setValue('');
  }

  async analyzeSource(source: RadianceModelSource) {
    this.selectedSourceId.set(source.source_id);
    const run = await this.radianceRunService.startRun(source.source_id);
    this.statusMessage.set(`Started theoretical structural analysis for ${source.display_name}.`);
    const completedRun = await this.radianceRunService.waitForCompletion(run.runId);
    if (completedRun.status !== 'succeeded') {
      this.statusMessage.set(completedRun.statusMessage || `Run ${completedRun.runId} did not complete successfully.`);
      return;
    }

    const graphCollections = await this.radianceRunService.loadRunGraph(completedRun.runId);
    this.modelLoaderService.loadedGraphCollections.set(graphCollections);
    this.statusMessage.set(`Loaded graph for ${completedRun.displayName}.`);
  }

  async reopenRun(runId: string) {
    const graphCollections = await this.radianceRunService.loadRunGraph(runId);
    this.modelLoaderService.loadedGraphCollections.set(graphCollections);
    const run = this.radianceRunService.runs().find((item) => item.runId === runId);
    this.statusMessage.set(run ? `Reopened ${run.displayName}.` : `Reopened ${runId}.`);
  }

  async deleteRun(runId: string) {
    const run = this.radianceRunService.runs().find((item) => item.runId === runId);
    await this.radianceRunService.deleteRun(runId);
    this.statusMessage.set(run ? `Deleted ${run.displayName}.` : `Deleted ${runId}.`);
  }

  formatLocator(source: RadianceModelSource): string {
    switch (source.kind) {
      case 'mounted_path':
        return source.mounted_subpath || '';
      case 'hugging_face':
        return source.huggingface_revision
          ? `${source.huggingface_repo} @ ${source.huggingface_revision}`
          : source.huggingface_repo || '';
      case 'staged_upload':
        return source.upload_filename || '';
      default:
        return source.display_name;
    }
  }

  sourceKindLabel(source: RadianceModelSource): string {
    switch (source.kind) {
      case 'mounted_path':
        return 'Mounted path';
      case 'hugging_face':
        return 'Hugging Face';
      case 'staged_upload':
        return 'Staged upload';
      default:
        return source.kind;
    }
  }

  canAnalyze(source: RadianceModelSource): boolean {
    return !this.submitting() && !this.submittingRun() && this.pollingRunId() == null;
  }

  canDeleteRun(runId: string, status: string): boolean {
    return (
      !this.submittingRun() &&
      this.pollingRunId() == null &&
      runId !== this.pollingRunId() &&
      status !== 'pending' &&
      status !== 'running'
    );
  }

  private async registerMountedSource(mountedSubpath: string) {
    const modelSource = await this.radianceSourceService.addMountedSource(
      mountedSubpath,
      this.mountedDisplayName.value?.trim() || '',
    );
    this.statusMessage.set(`Registered mounted source ${modelSource.display_name}.`);
  }
}
