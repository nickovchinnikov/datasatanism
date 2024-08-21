import { FC, useState } from "react";

import { useInterval } from "@/components/hooks"

import styles from "./Init.module.scss";

interface Props {
    init?: number;
    max?: number;
    speed?: number | null;
}

export const Loading: FC<Props> = ({ init = 0, max = 24 }) => {
    const unloadedCharacter = ".";
    const loadedCharacter = "#";

    const loaded = new Array(init).fill(loadedCharacter);
    const unloaded = new Array(max - init).fill(unloadedCharacter);
    const text = [...loaded, ...unloaded].join("")

    return <>
        <p className={styles.loadingBar}>{text}</p>
        <p className={styles.textSm}>PROCESS: <span className="process-amount">{Math.round(100 * init / max)}</span>%</p>
    </>
}