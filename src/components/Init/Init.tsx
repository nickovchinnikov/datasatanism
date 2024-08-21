import { FC } from "react";

import styles from "./Init.module.scss";

import { Terminal } from "./Terminal";
import { Spinner } from "./Spinner";
import { Loading } from "./Loading";

interface Props {
  glitch: boolean;
  systemMsg?: string;
  systemInf?: string;
  success?: boolean;
  successMsg?: string;
  loading: {
    init?: number;
    max?: number;
    speed?: number | null;
  }
}

export const Init: FC<Props> = ({
    glitch,
    systemMsg = "< SYSTEM REBOOTING >",
    systemInf = "HYDRA VER 2.1 SYS RECOVERY",
    success = false,
    successMsg = "REBOOTING SUCCESSFUL",
    loading: { init = 0, max = 24, speed = 250 } 
}) => (
    <Terminal glitch={glitch}>
        <div className={styles.scanline}></div>
        <Spinner isActive={glitch} />
        <div className={`${styles.hydra} ${success && styles.anim}`}>
            <div className={styles.hydra_rebooting}>
                {!success && <>
                    <p>{systemMsg}</p>
                    <p className={styles.textSm}>{systemInf}</p>
                    <Loading init={init} max={max} speed={speed} />
                </>}
                <div className={`${!success && styles.hidden}`}>
                    <p>{successMsg}</p>
                </div>
            </div>
        </div>            
    </Terminal>
)
