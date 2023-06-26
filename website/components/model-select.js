import { createSelectV } from './select.js'

export function createModelSelectV(label, modelsMap, any) {
    const modelNames = [...Object.keys(modelsMap)]
    const modelNameToShortName = Object.fromEntries(modelNames.map(modelName =>
        [modelName, modelsMap[modelName].short_name ?? modelsMap[modelName].model_name]))
    const shortNames = [...Object.values(modelNameToShortName)]
    const sortedShortNames = shortNames.toSorted((a, b) => a.localeCompare(b))
    const shortNameToModelName = Object.fromEntries(Object.entries(modelNameToShortName).map(([k, v]) => [v, k]))
    const sortedModelNames = sortedShortNames.map(shortName => shortNameToModelName[shortName])
    const ids = sortedModelNames.map(modelName => modelName.replace('/', '--'))
    if (any)
        return createSelectV(label, ['any'].concat(sortedShortNames), ['any'].concat(ids))
    return createSelectV(label, sortedShortNames, ids)
}
