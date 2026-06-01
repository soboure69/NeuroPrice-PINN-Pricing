import { NextResponse } from "next/server";

export async function GET() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  const adminSecret = process.env.ADMIN_API_SECRET;
  if (!apiUrl || !adminSecret) {
    return NextResponse.json({ error: "Admin dashboard is not configured." }, { status: 500 });
  }

  const response = await fetch(`${apiUrl.replace(/\/$/, "")}/api/v1/admin/summary`, {
    headers: {
      "X-NeuroPrice-Admin-Secret": adminSecret,
    },
    cache: "no-store",
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    return NextResponse.json({ error: body?.detail ?? "Admin summary request failed." }, { status: response.status });
  }
  return NextResponse.json(body);
}
