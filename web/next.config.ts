import type { NextConfig } from "next";

const backend = process.env.BACKEND_INTERNAL_URL ?? "http://localhost:8000";

// On Vercel this app is deployed as static/served client-side and talks to the
// model backend directly via NEXT_PUBLIC_API_URL, so no standalone bundle and
// no rewrite are wanted there. Standalone is only for self-hosted/Docker.
const ON_VERCEL = process.env.VERCEL === "1";

const nextConfig: NextConfig = {
  ...(ON_VERCEL ? {} : { output: "standalone" }),
  async rewrites() {
    // Only rewrite /api/* to a same-origin backend when not on Vercel.
    if (ON_VERCEL) return [];
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
