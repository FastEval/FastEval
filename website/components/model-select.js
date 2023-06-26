import { createSelectV } from './select.js'

export function createModelSelectV(label, modelsMap, any) {
    const modelNames = [...Object.keys(modelsMap)]
    const modelNamesOrShortNames = modelNames.map(modelName => modelsMap[modelName].short_name ?? modelsMap[modelName].model_name)
    const ids = modelNames.map(modelName => modelName.replace('/', '--'))
    if (any)
        return createSelectV(label, ['any'].concat(modelNamesOrShortNames), ['any'].concat(ids))
    return createSelectV(label, modelNamesOrShortNames, ids)
}
