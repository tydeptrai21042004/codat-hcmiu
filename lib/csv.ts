import Papa from "papaparse";
import { CLASS_CODES } from "./constants";
import type { DxCode, HamMetadataRow } from "./types";

function normalizeKey(row: Record<string, unknown>, key: string): string {
  const raw = row[key];
  return typeof raw === "string" ? raw.trim() : raw == null ? "" : String(raw).trim();
}

export async function parseMetadataCsv(file: File): Promise<HamMetadataRow[]> {
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, unknown>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        const rows = result.data
          .map((row) => {
            const lesion_id = normalizeKey(row, "lesion_id");
            const image_id = normalizeKey(row, "image_id").replace(/\.(jpg|jpeg|png)$/i, "");
            const dx = normalizeKey(row, "dx") as DxCode;
            if (!lesion_id || !image_id || !CLASS_CODES.includes(dx)) return null;
            return {
              lesion_id,
              image_id,
              dx,
              dx_type: normalizeKey(row, "dx_type"),
              age: normalizeKey(row, "age"),
              sex: normalizeKey(row, "sex"),
              localization: normalizeKey(row, "localization")
            } satisfies HamMetadataRow;
          })
          .filter(Boolean) as HamMetadataRow[];
        resolve(rows);
      },
      error: (err) => reject(err)
    });
  });
}

export function rowsToCsv(rows: Array<Record<string, unknown>>): string {
  return Papa.unparse(rows);
}
