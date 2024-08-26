import { FC } from 'react';

import { ManageModel, ModelState } from '@/utils/types';
import { useWllama, WllamaProvider } from '@/utils/wllama.context';

const ModelCard: FC<{
    model: ManageModel;
}> = ({ model }) => {
    const {
    downloadModel,
    removeModel,
    loadModel,
    unloadModel,
    removeCustomModel,
    currRuntimeInfo,
  } = useWllama();

//   const m = model;
    const percent = parseInt(Math.round(model.downloadPercent * 100).toString());
    console.log("model: ", model);
    console.log("percent: ", percent);
    
    return <h1>ModelCard: {model.name}</h1>
};

export const Models: FC = () => {
    const {
        models,
        removeModel,
        isLoadingModel,
        isDownloading,
        currModel,
        currParams,
        setParams,
      } = useWllama();

      console.log("models>: ", models);

      return models
        // .filter((m) => m.userAdded)
        .map((m) => (
          <ModelCard key={m.url} model={m} />
        ))
}