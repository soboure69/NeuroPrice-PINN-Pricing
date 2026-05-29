import posthog from "posthog-js";

type AnalyticsProperties = Record<string, string | number | boolean | null | undefined>;

export function identifyUser(email: string, properties: AnalyticsProperties = {}) {
  if (!process.env.NEXT_PUBLIC_POSTHOG_KEY) {
    return;
  }
  posthog.identify(email, properties);
}

export function resetAnalytics() {
  if (!process.env.NEXT_PUBLIC_POSTHOG_KEY) {
    return;
  }
  posthog.reset();
}

export function trackEvent(event: string, properties: AnalyticsProperties = {}) {
  if (!process.env.NEXT_PUBLIC_POSTHOG_KEY) {
    return;
  }
  posthog.capture(event, properties);
}
