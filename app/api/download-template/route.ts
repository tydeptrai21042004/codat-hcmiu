const template = [
  "lesion_id,image_id,dx,dx_type,age,sex,localization",
  "HAM_demo_0001,ISIC_0000001,nv,histo,45,male,back",
  "HAM_demo_0002,ISIC_0000002,mel,histo,66,female,lower extremity",
  "HAM_demo_0003,ISIC_0000003,bkl,consensus,53,male,trunk"
].join("\n");

export function GET() {
  return new Response(template, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": "attachment; filename=HAM10000_metadata_template.csv"
    }
  });
}
