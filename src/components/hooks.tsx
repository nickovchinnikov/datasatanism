import { useEffect, useRef, EffectCallback } from 'react';

export function useInterval(callback: EffectCallback, delay: number | null = 300) {
    const savedCallback = useRef(callback);

    useEffect(() => {
        savedCallback.current = callback;
    }, [callback]);

    useEffect(() => {
        function func() {
            savedCallback.current();
        }
        if (delay !== null) {
            const id = setInterval(func, delay);
            return () => clearInterval(id);
        }
    }, [delay]);
}
