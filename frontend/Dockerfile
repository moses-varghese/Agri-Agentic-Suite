# Stage 1: Build the Next.js application
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
RUN npm install mic-recorder-to-mp3 react-markdown
COPY . .
# Make build-time env vars available
ARG NEXT_PUBLIC_GROUNDTRUTH_AI_URL
ENV NEXT_PUBLIC_GROUNDTRUTH_AI_URL=$NEXT_PUBLIC_GROUNDTRUTH_AI_URL
ARG NEXT_PUBLIC_GROUNDTRUTH_AI_VOICE_URL
ENV NEXT_PUBLIC_GROUNDTRUTH_AI_VOICE_URL=$NEXT_PUBLIC_GROUNDTRUTH_AI_VOICE_URL
ARG NEXT_PUBLIC_YIELDWISE_URL
ENV NEXT_PUBLIC_YIELDWISE_URL=$NEXT_PUBLIC_YIELDWISE_URL
ARG NEXT_PUBLIC_FIELDSCOUT_AI_URL
ENV NEXT_PUBLIC_FIELDSCOUT_AI_URL=$NEXT_PUBLIC_FIELDSCOUT_AI_URL
RUN npm run build

# Stage 2: Serve the application
FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
ENV PORT 3000

CMD ["node", "server.js"]