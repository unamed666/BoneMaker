bl_info = {
    "name": "Delete Empty Vertex Groups",
    "blender": (3, 0, 0),
    "category": "Object",
}

import bpy

class OBJECT_OT_delete_empty_vgroups(bpy.types.Operator):
    """Hapus semua vertex group yang tidak memiliki vertex"""
    bl_idname = "object.delete_empty_vgroups"
    bl_label = "Delete Empty VGroups"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Pilih objek mesh terlebih dahulu")
            return {'CANCELLED'}
        
        vgs = obj.vertex_groups
        to_delete = []

        # Cek group kosong
        for vg in vgs:
            found = False
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group == vg.index:
                        found = True
                        break
                if found:
                    break
            if not found:
                to_delete.append(vg)
        
        # Hapus group kosong
        for vg in to_delete:
            vgs.remove(vg)

        self.report({'INFO'}, f"Dihapus {len(to_delete)} vertex group kosong")
        return {'FINISHED'}


class OBJECT_PT_vgroup_tools(bpy.types.Panel):
    bl_label = "Vertex Group Tools"
    bl_idname = "OBJECT_PT_vgroup_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BoneMaker'

    def draw(self, context):
        layout = self.layout
        layout.operator("object.delete_empty_vgroups", icon='TRASH')


def register():
    bpy.utils.register_class(OBJECT_OT_delete_empty_vgroups)
    bpy.utils.register_class(OBJECT_PT_vgroup_tools)

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_delete_empty_vgroups)
    bpy.utils.unregister_class(OBJECT_PT_vgroup_tools)

if __name__ == "__main__":
    register()
