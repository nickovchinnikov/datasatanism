import { FC, MouseEvent, ReactNode } from "react";

import styles from "./GlitchButton.module.scss";

interface Props {
    children?: ReactNode
    textDecoration?: string
    decoration?: "&rArr;" | "&lArr;" | "&uArr;" | "&dArr;"
    size?: number
    onClick?: (e: MouseEvent<HTMLAnchorElement>) => void
}

export const GlitchButton: FC<Props> = ({
    children = "Download",
    textDecoration = "_",
    decoration = "&rArr;",
    size = 1.5,
    onClick = () => {},
}) => <div className={styles.centerCenter} style={{ fontSize: `${size}rem` }}>
    <a className={styles.btnGlitch} onClick={onClick}>
        <span className={styles.decoration}
            dangerouslySetInnerHTML={{ __html: decoration }}
            style={{ fontSize: `${size * 1.37}rem` }}
        />
        <span className={styles.text}> {children}</span>
        <span className={styles.textDecoration}>{textDecoration}</span>
    </a>
</div>
