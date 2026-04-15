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

export interface RadianceModelSource {
  source_id: string;
  kind: string;
  display_name: string;
  mounted_subpath?: string | null;
  huggingface_repo?: string | null;
  huggingface_revision?: string | null;
  upload_filename?: string | null;
  trust_remote_code: boolean;
}

export interface MountedBrowserEntry {
  entryType: 'directory' | 'file';
  name: string;
  subpath: string;
}

export interface MountedBrowserPayload {
  currentSubpath: string;
  parentSubpath?: string | null;
  entries: MountedBrowserEntry[];
}

interface ModelSourceListResponse {
  sources: RadianceModelSource[];
}

interface CreateModelSourceResponse {
  modelSource: RadianceModelSource;
  authAccepted?: boolean;
}

const MODEL_SOURCES_API_PATH = '/api/v1/radiance/model-sources';
const MOUNTED_BROWSER_API_PATH = '/api/v1/radiance/model-sources/mounted';
const MOUNTED_SELECTION_API_PATH =
  '/api/v1/radiance/model-sources/mounted/selection';
const HUGGING_FACE_API_PATH = '/api/v1/radiance/model-sources/huggingface';
const UPLOAD_API_PATH = '/api/v1/radiance/model-sources/upload';

@Injectable({
  providedIn: 'root',
})
export class RadianceSourceService {
  readonly loadingSources = signal<boolean>(false);
  readonly loadingMountedEntries = signal<boolean>(false);
  readonly submitting = signal<boolean>(false);
  readonly lastError = signal<string | null>(null);
  readonly sources = signal<RadianceModelSource[]>([]);
  readonly mountedBrowser = signal<MountedBrowserPayload>({
    currentSubpath: '',
    parentSubpath: null,
    entries: [],
  });

  constructor() {
    void this.refreshSources();
    void this.browseMounted('');
  }

  async refreshSources() {
    this.loadingSources.set(true);
    try {
      const payload = await this.request<ModelSourceListResponse>(
        MODEL_SOURCES_API_PATH,
      );
      this.sources.set(payload.sources);
    } finally {
      this.loadingSources.set(false);
    }
  }

  async browseMounted(subpath = '') {
    this.loadingMountedEntries.set(true);
    try {
      const url = subpath
        ? `${MOUNTED_BROWSER_API_PATH}?subpath=${encodeURIComponent(subpath)}`
        : MOUNTED_BROWSER_API_PATH;
      const payload = await this.request<MountedBrowserPayload>(url);
      this.mountedBrowser.set(payload);
    } finally {
      this.loadingMountedEntries.set(false);
    }
  }

  async addMountedSource(mountedSubpath: string, displayName: string) {
    this.submitting.set(true);
    try {
      const payload = await this.request<CreateModelSourceResponse>(
        MOUNTED_SELECTION_API_PATH,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            mountedSubpath,
            displayName,
          }),
        },
      );
      await this.refreshSources();
      return payload.modelSource;
    } finally {
      this.submitting.set(false);
    }
  }

  async addHuggingFaceSource(payload: {
    repoId: string;
    revision: string;
    displayName: string;
    accessToken: string;
    trustRemoteCode: boolean;
  }) {
    this.submitting.set(true);
    try {
      const response = await this.request<CreateModelSourceResponse>(
        HUGGING_FACE_API_PATH,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        },
      );
      await this.refreshSources();
      return response;
    } finally {
      this.submitting.set(false);
    }
  }

  async uploadModel(file: File) {
    this.submitting.set(true);
    try {
      const formData = new FormData();
      formData.append('file', file, file.name);
      const payload = await this.request<CreateModelSourceResponse>(
        UPLOAD_API_PATH,
        {
          method: 'POST',
          body: formData,
        },
      );
      await this.refreshSources();
      return payload.modelSource;
    } finally {
      this.submitting.set(false);
    }
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
