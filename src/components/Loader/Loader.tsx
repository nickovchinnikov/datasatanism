import { FC } from "react";

import styles from "./Loader.module.scss";

interface Props {
  width?: string;
  height?: string;
}

export const Loader: FC<Props> = ({ width = "100vw", height = "100vh" }) => (
  <div className={styles.loader} style={{ width, height }}>
    <svg className={styles.circleFW} viewBox="0 0 1000 1000">
      <circle
        className={styles.path}
        cx="500"
        cy="500"
        fill="none"
        r="355"
        stroke="#29B6F6"
      />
    </svg>
    <svg
      className={styles.circleSW}
      style={{ animationDuration: "1.4s" }}
      viewBox="0 0 1000 1000"
    >
      <circle
        className={styles.path2}
        cx="500"
        cy="500"
        fill="none"
        r="355"
        stroke="#18FFFF"
      />
    </svg>
    <svg className={styles.circleFW} viewBox="0 0 1000 1000">
      <circle
        className={styles.path3}
        cx="500"
        cy="500"
        fill="none"
        r="355"
        stroke="#18FFFF"
      />
    </svg>
    <svg className={styles.circleFW} viewBox="0 0 1000 1000">
      <circle
        className={styles.path4}
        cx="500"
        cy="500"
        fill="none"
        r="255"
        stroke="#FFF"
      />
    </svg>
    <svg className={styles.circleFW} viewBox="0 0 1000 1000">
      <circle
        className={styles.path5}
        cx="500"
        cy="500"
        fill="none"
        r="420"
        stroke="#18FFFF"
      />
    </svg>
    <svg className={styles.circleFW} viewBox="0 0 1000 1000">
      <circle
        className={styles.path6}
        cx="500"
        cy="500"
        fill="none"
        r="420"
        stroke="#18FFFF"
      />
    </svg>
    <svg className={styles.circleSW} viewBox="0 0 1000 1000">
      <circle
        className={styles.path7}
        cx="500"
        cy="500"
        fill="none"
        r="420"
        stroke="#18FFFF"
      />
    </svg>
    <svg
      className={styles.circleSW}
      style={{ animationTimingFunction: "ease-in-out" }}
      viewBox="0 0 1000 1000"
    >
      <circle
        className={styles.path8}
        cx="500"
        cy="500"
        fill="none"
        r="420"
        stroke="#18FFFF"
      />
      <svg className={styles.circleFW} viewBox="0 0 1000 1000"></svg>
      <circle
        className={styles.path9}
        cx="500"
        cy="500"
        fill="none"
        r="225"
        stroke="#18FFFF"
      />
    </svg>
  </div>
);
