import Anthropic from '@anthropic-ai/sdk';
import 'dotenv/config';
import { VoyageAIClient } from 'voyageai';

const voyageClient = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY });

const anthropic = new Anthropic();

export async function sendMessage(threadId: string, content: { text: string; url?: string }) {
    console.log('sendMessage', threadId, JSON.stringify(content));
    //Set default response
    let response = { content: 'Sorry, I am not able to understand your question. Please try again.' };

    return response;
}

export async function embed(text: string) {
    return (
        await voyageClient.embed({
            input: text,
            model: 'voyage-3-lite',
        })
    ).data;
}
