import { createSelectV } from './select.js'

export function createModelSelectV(label, modelNames) {
    return createSelectV(label, modelNames, modelNames.map(modelName => modelName.replace('/', '--')))
}
