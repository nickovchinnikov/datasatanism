import { FC, useEffect, useReducer } from "react";
import { createSlice, bindActionCreators, Dispatch } from '@reduxjs/toolkit';

import { typeString } from "@/components/utils";
import { Init } from "@/components/Init/Init"

const initialState = {
    glitch: true,
    success: false,
    loading: { init: 0, max: 24, speed: null}
}

const loadingSlice = createSlice({
    name: 'loading',
    initialState,
    reducers: {
        toggle: (state) => {
            state.glitch = !state.glitch
            state.success = !state.success
        },
        loadingInc: (state) => {
            const next = state.loading.init + 1;
            return {
                ...state,
                loading: {
                    ...state.loading,
                    init: next < state.loading.max ? next : state.loading.max
                } 
            }
        }
    }
})

export const { actions, reducer, getInitialState } = loadingSlice;

interface Props {
    systemMsg?: string;
    systemInf?: string;
    successMsg?: string;
}

export const Loading: FC<Props> = ({
    systemMsg = "Download and install the model: ",
    systemInf = "LLAMA 3.1 8B params 7b quantized",
    successMsg = "Installed: LLAMA 3.1 8B"
}) => {
    const [state, dispatch] = useReducer(reducer, getInitialState())
    const { toggle, loadingInc } = bindActionCreators(actions, dispatch as Dispatch)
    const unloadedCharacter = ".";

    useEffect(() => {
        const line = (new Array(state.loading.max).fill(unloadedCharacter)).join("");
        if (line) {
          (async () => {
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            for await (const c of typeString(line)) {
                loadingInc();
            }
            toggle();
          })();
        }
      }, []);

    return <Init
        systemMsg={systemMsg}
        systemInf={systemInf}
        successMsg={successMsg}
        glitch={state.glitch}
        success={state.success}
        loading={state.loading}
    />
}