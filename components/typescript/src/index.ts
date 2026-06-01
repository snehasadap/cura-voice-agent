import "dotenv/config";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { createAgent, AIMessage, ToolMessage } from "langchain";
import path from "node:path";
import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { cors } from "hono/cors";
import { createNodeWebSocket } from "@hono/node-ws";
import type { WSContext } from "hono/ws";
import type WebSocket from "ws";
import { iife, writableIterator } from "./utils";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { v4 as uuidv4 } from "uuid";
import { CARTESIA_TTS_SYSTEM_PROMPT, CartesiaTTS } from "./cartesia";
import { AssemblyAISTT } from "./assemblyai/index";
import type { VoiceAgentEvent } from "./types";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const STATIC_DIR = path.join(__dirname, "../../web/dist");
const PORT = parseInt(process.env.PORT ?? "8000");

if (!existsSync(STATIC_DIR)) {
  console.error(
    `Web build not found at ${STATIC_DIR}.\n` +
      "Run 'make build-web' or 'make dev-ts' from the project root."
  );
  process.exit(1);
}

const app = new Hono();
const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });

app.use("/*", cors());

// ---------------------------------------------------------------------------
// In-memory appointment store — persists for the lifetime of the server
// ---------------------------------------------------------------------------
const ALL_SLOTS = ["9:00 AM", "10:30 AM", "1:00 PM", "2:30 PM", "4:00 PM"];

interface Appointment {
  patientName: string;
  specialty: string;
  date: string;
  time: string;
  confirmationCode: string;
}

const appointments = new Map<string, Appointment>();

function bookedSlots(specialty: string, date: string): Set<string> {
  const taken = new Set<string>();
  for (const a of appointments.values()) {
    if (a.specialty.toLowerCase() === specialty.toLowerCase() &&
        a.date.toLowerCase() === date.toLowerCase()) {
      taken.add(a.time);
    }
  }
  return taken;
}

const checkAvailability = tool(
  async ({ specialty, preferredDate }) => {
    const taken = bookedSlots(specialty, preferredDate);
    const available = ALL_SLOTS.filter(s => !taken.has(s));
    if (available.length === 0)
      return `No available slots for ${specialty} on ${preferredDate}. Please try another date.`;
    return `Available slots for ${specialty} on ${preferredDate}: ${available.join(", ")}.`;
  },
  {
    name: "check_availability",
    description: "Check available appointment slots for a given specialty and preferred date.",
    schema: z.object({
      specialty: z.string().describe("The medical specialty or type of doctor"),
      preferredDate: z.string().describe("The patient's preferred date (e.g. Monday, June 2nd)"),
    }),
  }
);

const bookAppointment = tool(
  async ({ patientName, specialty, date, time }) => {
    if (bookedSlots(specialty, date).has(time))
      return `${time} on ${date} is already taken for ${specialty}. Please choose another slot.`;
    const code = `HC${Math.floor(10000 + Math.random() * 90000)}`;
    appointments.set(code, { patientName, specialty, date, time, confirmationCode: code });
    return `Appointment booked for ${patientName} with ${specialty} on ${date} at ${time}. Confirmation code: ${code}.`;
  },
  {
    name: "book_appointment",
    description: "Book an appointment for the patient once they have chosen a slot.",
    schema: z.object({
      patientName: z.string().describe("Full name of the patient"),
      specialty: z.string().describe("The medical specialty"),
      date: z.string().describe("Appointment date"),
      time: z.string().describe("Appointment time"),
    }),
  }
);

const listAppointments = tool(
  async ({ patientName }) => {
    const appts = [...appointments.values()].filter(
      a => a.patientName.toLowerCase() === patientName.toLowerCase()
    );
    if (appts.length === 0) return `No appointments found for ${patientName}.`;
    const lines = appts.map(
      a => `${a.specialty} on ${a.date} at ${a.time} — confirmation ${a.confirmationCode}`
    );
    return `Appointments for ${patientName}: ${lines.join("; ")}.`;
  },
  {
    name: "list_appointments",
    description: "List all appointments for a patient by name.",
    schema: z.object({
      patientName: z.string().describe("Full name of the patient"),
    }),
  }
);

const cancelAppointment = tool(
  async ({ confirmationCode }) => {
    const appt = appointments.get(confirmationCode.toUpperCase());
    if (!appt) return `No appointment found with confirmation code ${confirmationCode}.`;
    appointments.delete(confirmationCode.toUpperCase());
    return `Appointment ${confirmationCode} for ${appt.patientName} (${appt.specialty} on ${appt.date} at ${appt.time}) has been cancelled.`;
  },
  {
    name: "cancel_appointment",
    description: "Cancel an appointment by confirmation code.",
    schema: z.object({
      confirmationCode: z.string().describe("The confirmation code from the booking"),
    }),
  }
);

const systemPrompt = `
You are a friendly and professional healthcare receptionist. Your goal is to help patients book, view, and cancel medical appointments.
Be concise, empathetic, and clear.

Available specialties: general practice, cardiology, dermatology, orthopedics, pediatrics, neurology.
Available days: Monday through Friday.
Available hours: 9:00 AM to 5:00 PM in 90-minute slots.

To book an appointment collect: patient full name, specialty or reason for visit, preferred date and time.
Use check_availability to show open slots before booking.
Use book_appointment once the patient confirms a slot.
Use list_appointments when a patient wants to see their upcoming appointments.
Use cancel_appointment when a patient wants to cancel using their confirmation code.

${CARTESIA_TTS_SYSTEM_PROMPT}
`;

const agent = createAgent({
  model: "claude-haiku-4-5",
  tools: [checkAvailability, bookAppointment, listAppointments, cancelAppointment],
  checkpointer: new MemorySaver(),
  systemPrompt: systemPrompt,
});

/**
 * Transform stream: Audio (Uint8Array) → Voice Events (VoiceAgentEvent)
 *
 * This function takes a stream of audio chunks and sends them to AssemblyAI for STT.
 *
 * It uses a producer-consumer pattern where:
 * - Producer: Reads audio chunks from audioStream and sends them to AssemblyAI
 * - Consumer: Receives transcription events from AssemblyAI and yields them
 *
 * @param audioStream - Async iterator of PCM audio bytes (16-bit, mono, 16kHz)
 * @returns Async generator yielding STT events (stt_chunk for partials, stt_output for final transcripts)
 */
async function* sttStream(
  audioStream: AsyncIterable<Uint8Array>
): AsyncGenerator<VoiceAgentEvent> {
  const stt = new AssemblyAISTT({ sampleRate: 16000 });
  const passthrough = writableIterator<VoiceAgentEvent>();

  /**
   * Promise that pumps audio chunks to AssemblyAI.
   *
   * This runs concurrently with the consumer, continuously reading audio
   * chunks from the input stream and forwarding them to AssemblyAI.
   * This allows transcription to begin before all audio has arrived.
   */
  const producer = iife(async () => {
    try {
      // Stream each audio chunk to AssemblyAI as it arrives
      for await (const audioChunk of audioStream) {
        await stt.sendAudio(audioChunk);
      }
    } finally {
      // Signal to AssemblyAI that audio streaming is complete
      await stt.close();
    }
  });

  /**
   * Promise that receives transcription events from AssemblyAI.
   *
   * This runs concurrently with the producer, listening for STT events
   * and pushing them into the passthrough iterator for downstream stages.
   */
  const consumer = iife(async () => {
    for await (const event of stt.receiveEvents()) {
      passthrough.push(event);
    }
  });

  try {
    // Yield events as they arrive from the consumer
    yield* passthrough;
  } finally {
    // Wait for the producer and consumer to complete when cleaning up
    await Promise.all([producer, consumer]);
  }
}

/**
 * Transform stream: Voice Events → Voice Events (with Agent Responses)
 *
 * This function takes a stream of upstream voice agent events and processes them.
 * When an stt_output event arrives, it passes the transcript to the LangChain agent.
 * The agent streams back its response tokens as agent_chunk events.
 * Tool calls and results are also emitted as separate events.
 * All other upstream events are passed through unchanged.
 *
 * @param eventStream - An async iterator of upstream voice agent events
 * @returns Async generator yielding all upstream events plus agent_chunk, tool_call, and tool_result events
 */
async function* agentStream(
  eventStream: AsyncIterable<VoiceAgentEvent>
): AsyncGenerator<VoiceAgentEvent> {
  const threadId = uuidv4();
  const passthrough = writableIterator<VoiceAgentEvent>();

  let transcriptBuffer: string[] = [];
  let latestTranscript = "";
  let debounceTimer: NodeJS.Timeout | null = null;

  async function runAgent(transcript: string) {
    const stream = await agent.stream(
      { messages: [new HumanMessage(transcript)] },
      {
        configurable: { thread_id: threadId },
        streamMode: "messages",
      }
    );

    for await (const [message] of stream) {
      if (AIMessage.isInstance(message) && message.tool_calls) {
        passthrough.push({ type: "agent_chunk", text: message.text, ts: Date.now() });

        for (const toolCall of message.tool_calls) {
          passthrough.push({
            type: "tool_call",
            id: toolCall.id ?? uuidv4(),
            name: toolCall.name,
            args: toolCall.args,
            ts: Date.now(),
          });
        }
      }

      if (ToolMessage.isInstance(message)) {
        passthrough.push({
          type: "tool_result",
          toolCallId: message.tool_call_id ?? "",
          name: message.name ?? "unknown",
          result:
            typeof message.content === "string"
              ? message.content
              : JSON.stringify(message.content),
          ts: Date.now(),
        });
      }
    }

    passthrough.push({ type: "agent_end", ts: Date.now() });
  }

  const producer = iife(async () => {
    for await (const event of eventStream) {
      passthrough.push(event);

      // Accumulate all stt_output fragments — if an utterance is split across
      // multiple end-of-turn events they get joined before the agent sees them.
      if (event.type === "stt_output") {
        transcriptBuffer.push(event.transcript);
        latestTranscript = transcriptBuffer.join(" ");
      }

      // Reset debounce on any speech activity so mid-sentence fragments don't
      // fire the agent while the user is still talking.
      if (event.type === "stt_chunk" || event.type === "stt_output") {
        if (debounceTimer) clearTimeout(debounceTimer);
        if (latestTranscript) {
          debounceTimer = setTimeout(() => {
            const transcript = latestTranscript;
            transcriptBuffer = [];
            latestTranscript = "";
            void runAgent(transcript);
          }, 250);
        }
      }
    }
  });

  try {
    yield* passthrough;
  } finally {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    await producer;
  }
}
/**
 * Transform stream: Voice Events → Voice Events (with Audio)
 *
 * This function takes a stream of upstream voice agent events and processes them.
 * When agent_chunk events arrive, it sends the text to ElevenLabs for TTS synthesis.
 * Audio is streamed back as tts_chunk events as it's generated.
 * All upstream events are passed through unchanged.
 *
 * It uses a producer-consumer pattern where:
 * - Producer: Reads events from eventStream, passes them through, and sends agent text to ElevenLabs
 * - Consumer: Receives audio chunks from ElevenLabs and yields them as tts_chunk events
 *
 * @param eventStream - An async iterator of upstream voice agent events
 * @returns Async generator yielding all upstream events plus tts_chunk events for synthesized audio
 */
async function* ttsStream(
  eventStream: AsyncIterable<VoiceAgentEvent>
): AsyncGenerator<VoiceAgentEvent> {
  const tts = new CartesiaTTS({
    voiceId: "f6ff7c0c-e396-40a9-a70b-f7607edb6937",
  });
  const passthrough = writableIterator<VoiceAgentEvent>();

  /**
   * Promise that reads events from the upstream stream and sends text to Cartesia.
   *
   * This runs concurrently with the consumer, continuously reading events
   * from the upstream stream and forwarding agent text to Cartesia for synthesis.
   * All events are passed through to the downstream via the passthrough iterator.
   * This allows audio generation to begin before the agent has finished generating.
   */
  const producer = iife(async () => {
    try {
      let buffer: string[] = [];
      for await (const event of eventStream) {
        // Pass through all events to downstream consumers
        passthrough.push(event);
        // Send agent text chunks to Cartesia for synthesis
        if (event.type === "agent_chunk") {
          buffer.push(event.text);
        }
        // Send all buffered text to Cartesia for synthesis
        if (event.type === "agent_end") {
          await tts.sendText(buffer.join(""));
          buffer = [];
        }
      }
    } finally {
      // Signal to Cartesia that text sending is complete
      await tts.close();
    }
  });

  /**
   * Promise that receives audio events from Cartesia.
   *
   * This runs concurrently with the producer, listening for TTS audio chunks
   * and pushing them into the passthrough iterator for downstream stages.
   */
  const consumer = iife(async () => {
    for await (const event of tts.receiveEvents()) {
      passthrough.push(event);
    }
  });

  try {
    // Yield events as they arrive from both producer (upstream) and consumer (TTS)
    yield* passthrough;
  } finally {
    // Wait for the producer and consumer to complete when cleaning up
    await Promise.all([producer, consumer]);
  }
}

app.get("/*", serveStatic({ root: STATIC_DIR }));

app.get(
  "/ws",
  upgradeWebSocket(async () => {
    let currentSocket: WSContext<WebSocket> | undefined;

    // Create a writable stream for incoming WebSocket audio data
    const inputStream = writableIterator<Uint8Array>();

    // Define the voice processing pipeline as a chain of async generators
    // Audio -> STT events
    const transcriptEventStream = sttStream(inputStream);
    // STT events -> STT Events + Agent events
    const agentEventStream = agentStream(transcriptEventStream);
    // STT events + Agent events -> STT Events + Agent Events + TTS events
    const outputEventStream = ttsStream(agentEventStream);

    const flushPromise = iife(async () => {
      // Process all events from the pipeline, sending events back to the client
      for await (const event of outputEventStream) {
        currentSocket?.send(JSON.stringify(event));
      }
    });

    return {
      onOpen(_, ws) {
        currentSocket = ws;
      },
      onMessage(event) {
        // Push incoming audio data into the pipeline's input stream
        const data = event.data;
        if (Buffer.isBuffer(data)) {
          inputStream.push(new Uint8Array(data));
        } else if (data instanceof ArrayBuffer) {
          inputStream.push(new Uint8Array(data));
        }
      },
      async onClose() {
        // Signal end of stream when socket closes
        inputStream.cancel();
        await flushPromise;
      },
    };
  })
);

const server = serve({
  fetch: app.fetch,
  port: PORT,
});

injectWebSocket(server);

console.log(`Server is running on port ${PORT}`);
