import { Module } from '@nitrostack/core';
import { PacsTools } from './pacs.tools.js';

/**
 * PACS integration module -- Phase 3, item 1.
 *
 * Bridges the LangGraph clinical agent to a hospital picture archiving
 * and communication system. Currently backed by a simulated archive
 * (see pacs.tools.ts); the module boundary is what lets a real DICOMweb
 * client replace it later without the agent side changing.
 */
@Module({
  name: 'pacs',
  description: 'Prior-imaging lookup against the hospital PACS archive (currently simulated)',
  controllers: [PacsTools],
})
export class PacsModule {}
