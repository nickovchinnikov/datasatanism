import { FC, ReactNode } from "react";

import styles from "./Init.module.scss";

interface Props {
    children?: ReactNode;
    glitch: boolean;
}

export const Terminal: FC<Props> = ({ children, glitch }) => {
    const glitchClass = glitch && styles.glitch;

    const baseClasses = [styles.terminal, glitchClass].join(" ");

    const bottomClasses = (hidden: boolean) => [
        baseClasses,
        styles.glitchClone,
        styles.glitchBottom,
        hidden && styles.hidden
    ].join(" ");
    
    const topClasses = (hidden: boolean) => [
        baseClasses,
        styles.glitchClone,
        styles.glitchTop,
        hidden && styles.hidden
    ].join(" ");

    const Glitch: FC<{ hidden?: boolean }> = ({ hidden = false }) => <>
        <div className={bottomClasses(hidden)}>{children}</div>
        <div className={topClasses(hidden)}>{children}</div>
    </>

    return <div style={{  display: "grid" }}>
        <div className={baseClasses}>{children}</div>
        <Glitch hidden={!glitch} />
    </div>
};
