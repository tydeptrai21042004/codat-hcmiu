type VercelResponse = {
  setHeader: (key: string, value: string) => void;
  status: (code: number) => VercelResponse;
  send: (body: string) => void;
};

const template = [
  "lesion_id,image_id,dx,dx_type,age,sex,localization",
  "HAM_demo_0001,ISIC_0000001,nv,histo,45,male,back",
  "HAM_demo_0002,ISIC_0000002,mel,histo,66,female,lower extremity",
  "HAM_demo_0003,ISIC_0000003,bkl,consensus,53,male,trunk"
].join("\n");

export default function handler(_req: unknown, res: VercelResponse) {
  res.setHeader("Content-Type", "text/csv; charset=utf-8");
  res.setHeader("Content-Disposition", "attachment; filename=HAM10000_metadata_template.csv");
  res.status(200).send(template);
}
