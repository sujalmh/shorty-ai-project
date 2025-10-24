import axios from "axios";

const BASE_URL = (import.meta as any).env?.VITE_BACKEND_URL || "http://localhost:8000";

export const api = axios.create({
    baseURL: `${BASE_URL}/api`,
    headers: {
        "Content-Type": "application/json",
    },
});
