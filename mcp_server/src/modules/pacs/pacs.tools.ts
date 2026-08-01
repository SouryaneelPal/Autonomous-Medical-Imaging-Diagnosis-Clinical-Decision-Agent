import { ToolDecorator as Tool, ExecutionContext, z } from '@nitrostack/core';

/**
 * PACS bridge tools -- Phase 3, item 1.
 *
 * Exposes prior-imaging lookup to the LangGraph agent over MCP. The
 * console's "Prior Studies" panel has been empty because this system has
 * no PACS integration; this module is the seam where a real one will
 * eventually plug in.
 *
 * ---------------------------------------------------------------------
 * EVERY STUDY RETURNED HERE IS SIMULATED.
 *
 * There is no PACS behind this. The records below are fixtures. They are
 * marked `simulated: true` on every study, and the response carries a
 * top-level `dataSource: 'SIMULATED'` plus an explicit notice, because a
 * prior-study list is read as patient history: rendered beside a real
 * radiograph, fabricated priors are indistinguishable from genuine ones
 * at a glance, and "no priors on file" and "priors that were invented"
 * are very different clinical facts. The same discipline applies here as
 * to the synthetic RSNA data elsewhere in this project -- anything not
 * real says so, in the payload, not just in a comment.
 *
 * Replacing this with a real PACS means swapping the fixture lookup for a
 * DICOMweb QIDO-RS query and dropping the simulated markers -- the tool
 * contract stays the same.
 * ---------------------------------------------------------------------
 *
 * Logging note: this server speaks MCP over STDIO, where stdout IS the
 * protocol channel. A stray console.log corrupts the JSON-RPC stream and
 * breaks the session. All logging goes through ctx.logger, which writes
 * to stderr.
 */

/** One prior imaging study, shaped after the DICOM tags a PACS query returns. */
interface PriorStudy {
  studyInstanceUid: string;
  studyDate: string;        // YYYY-MM-DD
  modality: string;         // DICOM Modality (0008,0060)
  bodyPartExamined: string; // (0018,0015)
  viewPosition: string;     // (0018,5101)
  studyDescription: string;
  reportImpression: string;
  reportedBy: string;
  simulated: boolean;
}

/**
 * Fixtures keyed by patientId. Deterministic on purpose: a tool that
 * returned randomised history would make the console's prior-studies
 * panel change between refreshes for the same patient, and would make
 * this untestable.
 */
const SIMULATED_ARCHIVE: Record<string, PriorStudy[]> = {
  'P-80213-XX': [
    {
      studyInstanceUid: '1.2.826.0.1.3680043.8.498.10001',
      studyDate: '2026-01-09',
      modality: 'CR',
      bodyPartExamined: 'CHEST',
      viewPosition: 'PA',
      studyDescription: 'Chest radiograph, routine pre-operative screening',
      reportImpression: 'No acute cardiopulmonary process. Lungs clear bilaterally.',
      reportedBy: 'SIMULATED-RADIOLOGIST',
      simulated: true,
    },
    {
      studyInstanceUid: '1.2.826.0.1.3680043.8.498.10002',
      studyDate: '2026-07-02',
      modality: 'CR',
      bodyPartExamined: 'CHEST',
      viewPosition: 'AP',
      studyDescription: 'Chest radiograph, productive cough',
      reportImpression:
        'Mild haziness at the right lung base, nonspecific. No confirmed consolidation at this time. ' +
        'Recommend clinical correlation and follow-up imaging if symptoms persist.',
      reportedBy: 'SIMULATED-RADIOLOGIST',
      simulated: true,
    },
  ],
};

/** Returned for any patient with no fixture, rather than inventing history. */
const NO_PRIORS: PriorStudy[] = [];

export class PacsTools {
  @Tool({
    name: 'query_prior_studies',
    description:
      'Query the hospital PACS archive for a patient\'s prior imaging studies. Returns DICOM ' +
      'study metadata (study date, modality, view position, and the prior report impression) ' +
      'ordered oldest to newest, for comparison against the current radiograph. NOTE: this ' +
      'server is backed by a simulated archive -- every study is marked simulated:true and must ' +
      'not be treated as real patient history.',
    inputSchema: z.object({
      patientId: z
        .string()
        .min(1)
        .describe('Patient identifier to look up in the PACS archive, e.g. "P-80213-XX"'),
    }),
    examples: {
      request: { patientId: 'P-80213-XX' },
      response: {
        patientId: 'P-80213-XX',
        dataSource: 'SIMULATED',
        studyCount: 2,
        studies: [
          {
            studyInstanceUid: '1.2.826.0.1.3680043.8.498.10001',
            studyDate: '2026-01-09',
            modality: 'CR',
            bodyPartExamined: 'CHEST',
            viewPosition: 'PA',
            studyDescription: 'Chest radiograph, routine pre-operative screening',
            reportImpression: 'No acute cardiopulmonary process. Lungs clear bilaterally.',
            reportedBy: 'SIMULATED-RADIOLOGIST',
            simulated: true,
          },
        ],
      },
    },
  })
  async queryPriorStudies(input: { patientId: string }, ctx: ExecutionContext) {
    const patientId = String(input.patientId ?? '').trim();

    // ctx.logger, never console.log -- stdout carries the JSON-RPC frames.
    ctx.logger.info('PACS prior-study query received', { patientId });

    if (!patientId) {
      ctx.logger.warn('PACS query rejected: empty patientId');
      throw new Error('patientId must be a non-empty string.');
    }

    const studies = SIMULATED_ARCHIVE[patientId] ?? NO_PRIORS;

    // A miss is reported as zero priors, never as invented history: the
    // caller must be able to tell "nothing on file" from "here is a past".
    if (studies.length === 0) {
      ctx.logger.info('No prior studies on file for patient', { patientId });
    } else {
      ctx.logger.info('Returning prior studies', {
        patientId,
        studyCount: studies.length,
        dateRange: `${studies[0].studyDate}..${studies[studies.length - 1].studyDate}`,
      });
    }

    return {
      patientId,
      dataSource: 'SIMULATED',
      notice:
        'SIMULATED DATA -- these prior studies are fixtures from a stand-in PACS, not real ' +
        'patient history. Do not use for clinical decision-making.',
      studyCount: studies.length,
      studies,
      queriedAt: new Date().toISOString(),
    };
  }
}
