

async function main() {
  const res = await fetch("https://ai.1d.gg/healthz", {
    headers: {
      Authorization: `Bearer ${process.env.GATEWAY_AUTH_TOKEN}`
    }
  });
  console.log(res.headers.get("x-sam-ai-router"))
}

main().catch(() => process.exit(1)).then(() => process.exit(0))
