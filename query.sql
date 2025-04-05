CREATE TABLE documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- Adjust the dimension based on your embedding model
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);