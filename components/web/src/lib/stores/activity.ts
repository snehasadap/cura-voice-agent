import { writable } from "svelte/store";
import type { ActivityItem, LogEntry } from "../types";

const LIVE_ID = "stt-live";

function createActivityStore() {
  const { subscribe, update, set } = writable<ActivityItem[]>([]);

  let idCounter = 0;

  return {
    subscribe,

    add(
      type: ActivityItem["type"],
      label: string,
      text: string,
      args?: Record<string, unknown>
    ) {
      const item: ActivityItem = {
        id: `activity-${++idCounter}`,
        type,
        label,
        text,
        args,
        timestamp: new Date(),
      };
      update((items) => [item, ...items].slice(0, 50));
    },

    // Create or update the single in-progress transcript card.
    setLive(text: string) {
      update((items) => {
        const idx = items.findIndex((i) => i.id === LIVE_ID);
        const item: ActivityItem = {
          id: LIVE_ID,
          type: "stt",
          label: "Transcription",
          text,
          timestamp: new Date(),
        };
        if (idx !== -1) {
          const copy = [...items];
          copy[idx] = item;
          return copy;
        }
        return [item, ...items].slice(0, 50);
      });
    },

    // Lock the live card in as a permanent item once the agent fires.
    commitLive() {
      update((items) => {
        const idx = items.findIndex((i) => i.id === LIVE_ID);
        if (idx === -1) return items;
        const copy = [...items];
        copy[idx] = { ...copy[idx], id: `activity-${++idCounter}` };
        return copy;
      });
    },

    clear() {
      set([]);
    },
  };
}

function createLogStore() {
  const { subscribe, update, set } = writable<LogEntry[]>([]);

  let idCounter = 0;

  return {
    subscribe,

    log(message: string) {
      const entry: LogEntry = {
        id: `log-${++idCounter}`,
        message,
        timestamp: new Date(),
      };
      update((entries) => [...entries, entry].slice(-100)); // Keep max 100 entries
    },

    clear() {
      set([]);
    },
  };
}

export const activities = createActivityStore();
export const logs = createLogStore();
