import mongoose, { Schema } from 'mongoose';

export interface IDbMessage {
    thread_id: string;
    role: string;
    content: any[];
    url?: string;
    created_at?: Date;
}

const messageSchema = new mongoose.Schema<IDbMessage>({
    thread_id: { type: String, required: true },
    role: { type: String, required: true },
    content: { type: Schema.Types.Mixed },
    url: { type: String },
    created_at: { type: Date, default: Date.now },
});

export const Message = mongoose.model<IDbMessage>('Message', messageSchema);

export const getAllMessages = async (threadId: string) => {
    return Message.find({ thread_id: threadId }).sort({ created_at: 1 });
};

export const createMessage = async (message: IDbMessage) => {
    return Message.create(message);
};
