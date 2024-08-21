import { useState, FC } from "react";

import { useInterval } from "@/components/hooks"

import styles from "./Init.module.scss";

interface Props {
    isActive?: boolean;
    speed?: number
}

export const Spinner: FC<Props> = ({ isActive = true, speed = 250 }) => {
    const symbols = ['/', '-', '\\', '|'];

    const [idx, setIdx] = useState(0);

    useInterval(() => {
        setIdx((prevIndex) => (prevIndex + 1) % symbols.length)
    }, isActive ? speed : null);

    return <p className={styles.spinner}>[{symbols[idx]}]</p>
};
