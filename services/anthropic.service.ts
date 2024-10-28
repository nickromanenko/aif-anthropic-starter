import Anthropic from '@anthropic-ai/sdk';
import 'dotenv/config';
import { VoyageAIClient } from 'voyageai';
import { createMessage, getAllMessages } from '../models/message.model';
import { searchIndex } from './embedder.service';

const voyageClient = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY });

const anthropic = new Anthropic();

export async function sendMessage(threadId: string, content: { text: string; url?: string }) {
    console.log('sendMessage', threadId, JSON.stringify(content));
    //Set default response
    let responseContent = 'Sorry, I am not able to understand your question. Please try again.';

    // 1. Add user message to the database
    await createMessage({
        thread_id: threadId,
        role: 'user',
        content: [
            {
                type: 'text',
                text: content.text,
            },
        ],
    });
    // 2. Load history of messages from the database
    const dbMessages = await getAllMessages(threadId);
    const messages: any[] = dbMessages.map(message => ({
        role: message.role,
        content: message.content,
    }));

    const msg = await createAnthropicMessage(messages);
    console.log(msg);

    // 3. If the assistant message is a text, set the response content
    if (msg.stop_reason === 'tool_use') {
        const toolCall: any = msg.content.find((m: any) => m.type === 'tool_use');
        if (toolCall) {
            // console.log('toolCall', toolCall);
            messages.push({
                role: 'assistant',
                content: [toolCall],
            });

            await createMessage({
                thread_id: threadId,
                role: 'assistant',
                content: msg.content,
            });

            const context = await getContext(toolCall.input.question);
            const toolResult: any = {
                type: 'tool_result',
                tool_use_id: toolCall.id,
            };
            if (context.length) {
                toolResult.content = context;
            }
            const responseMessage = {
                role: 'user',
                content: [toolResult],
            };
            messages.push(responseMessage);

            await createMessage({
                thread_id: threadId,
                role: 'user',
                content: responseMessage.content,
            });
        }
        // If the model used the getContext tool, we need to provide the context
        const newMsg = await createAnthropicMessage(messages);
        console.log(newMsg);
        if (newMsg.content[0].type === 'text') {
            responseContent = newMsg.content[0].text;
            // 5. Add assistant message to the database
            await createMessage({
                thread_id: threadId,
                role: 'assistant',
                content: newMsg.content,
            });
        }
    } else {
        if (msg.content[0].type === 'text') {
            responseContent = msg.content[0].text;
        }
    }

    return {
        content: responseContent,
    };
}

async function createAnthropicMessage(messages: any[]) {
    const instructions =
        "You are a customer support assistant for TechEase Solutions, a company that provides comprehensive IT services to businesses. Your role is to assist our clients with their technical issues, answer questions about our services, and provide guidance on using our products effectively. Always respond in a friendly, professional manner, and ensure your explanations are clear and concise. If you're unable to resolve an issue immediately, reassure the customer that you will escalate the problem and follow up promptly. Your goal is to provide exceptional support and ensure customer satisfaction.";
    const msg = await anthropic.messages.create({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1000,
        temperature: 0,
        system: instructions,
        messages: messages,
        tools: [
            {
                name: 'getContext',
                description: "Always retrieve relevant context to answer a user's IT/software related question",
                input_schema: {
                    type: 'object',
                    properties: {
                        question: {
                            type: 'string',
                            description: "The user's question is about IT or software and requires additional context.",
                        },
                    },
                    required: ['question'],
                },
            },
        ],
    });

    return msg;
}

async function getContext(query: string) {
    const embeddingResult = await embed(query);
    const matches = (await searchIndex(embeddingResult[0].embedding)).matches;
    return matches.length ? matches.map(match => match.metadata.content).join('\n\n') : '';
}

export async function embed(text: string) {
    return (
        await voyageClient.embed({
            input: text,
            model: 'voyage-3-lite',
        })
    ).data;
}
