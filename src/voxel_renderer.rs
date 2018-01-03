
use graphics::*;
use renderer::*;
use graphics_util::*;
use math::*;
use std::collections::HashMap;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::rc::*;
use std::cell::{RefCell,Cell};
use std::borrow::BorrowMut;
use std::cell::RefMut;

pub enum RenderLifetime{
    Manual,
    OneDraw,
}

pub enum RenderTransform{
    UI,
    World,
    None,
}

#[derive(Clone, Copy)]
pub enum RenderID{
    ID(usize),
}

impl PartialEq for RenderID{
    fn eq(&self, other: &RenderID) -> bool {
        match self{
            &RenderID::ID(x) => {
                match other{
                    &RenderID::ID(y) => x == y
                }
            }
        }
    }
}
impl Eq for RenderID{}

impl Hash for RenderID{
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self{
            &RenderID::ID(x) => {
                state.write_usize(x);
                state.finish();
            }
        }
    }
}

pub struct RenderDataProvider{
    pub pre_render_state: Option<Box<Fn()->()>>,
    pub post_render_state: Option<Box<Fn()->()>>,
    pub shader_data: Option<Box<Fn(&Program, &WindowInfo)->()>>,
}

pub struct RenderInfo{
    pub renderer: Box<RendererVertFrag>,
    pub provider: RenderDataProvider,
}

pub struct VoxelRenderer<'a>{
    pub lifetime_one_draw_renderers: HashMap<RenderID, RenderInfo>,
    pub lifetime_manual_renderers: HashMap<RenderID, RenderInfo>,

    shaders: &'a HashMap<String, Program>,
    render_id_counter: Cell<usize>,
}


impl<'a> VoxelRenderer<'a>{
    pub fn new(shaders: &'a HashMap<String, Program>) -> VoxelRenderer<'a>{
        VoxelRenderer{
            lifetime_one_draw_renderers: HashMap::new(),
            lifetime_manual_renderers: HashMap::new(),
            shaders,
            render_id_counter: Cell::new(0),
        }
    }


    pub fn manual_mut(&mut self, id : &RenderID) -> &mut Box<RendererVertFrag>{
        self.lifetime_manual_renderers.get_mut(id).unwrap().renderer.borrow_mut() as &mut Box<RendererVertFrag>
    }


    pub fn draw(&mut self, win_info: &WindowInfo){
        for render_info in self.lifetime_one_draw_renderers.values_mut(){
            let shader_name = &render_info.renderer.shader_name();
            let shader = self.shaders.get(shader_name).unwrap();
            shader.enable();

            if render_info.provider.pre_render_state.is_some(){
                let ref opt = render_info.provider.pre_render_state;
                match opt{
                    &Some(ref x) => (*x)(),
                    _ => (),
                }

            }

            if render_info.provider.shader_data.is_some(){
                let ref opt = render_info.provider.shader_data;
                match opt{
                    &Some(ref x) => (*x)(shader, win_info),
                    _ => (),
                }

            }


            render_info.renderer.construct();
            render_info.renderer.draw();
            render_info.renderer.deconstruct();


            if render_info.provider.post_render_state.is_some(){
                let ref opt = render_info.provider.post_render_state;
                match opt{
                    &Some(ref x) => (*x)(),
                    _ => (),
                }

            }

            shader.disable();

        }


        for render_info in self.lifetime_manual_renderers.values_mut(){
            let shader_name = &render_info.renderer.shader_name();
            let shader = self.shaders.get(shader_name).unwrap();
            shader.enable();

            if render_info.provider.pre_render_state.is_some(){
                let ref opt = render_info.provider.pre_render_state;
                match opt{
                    &Some(ref x) => (*x)(),
                    _ => (),
                }

            }

            if render_info.provider.shader_data.is_some(){
                let ref opt = render_info.provider.shader_data;
                match opt{
                    &Some(ref x) => (*x)(shader, win_info),
                    _ => (),
                }

            }

            //manual construction and deconstruction !
            render_info.renderer.draw();


            if render_info.provider.post_render_state.is_some(){
                let ref opt = render_info.provider.post_render_state;
                match opt{
                    &Some(ref x) => (*x)(),
                    _ => (),
                }

            }

            shader.disable();

        }
    }

    pub fn push(&mut self, life: RenderLifetime, trans: RenderTransform, mut renderer: RenderInfo) -> Result<RenderID, &'static str>{
        if !self.shaders.contains_key(&renderer.renderer.shader_name()) {return Err("error: shader not found")}



        fn provide_def(shader: &Program, win: &WindowInfo){ //shader will be already enabled
            shader.set_float4x4("P",  false, &[
                1.0,0.0,0.0,0.0,
                0.0,1.0,0.0,0.0,
                0.0,0.0,1.0,0.0,
                0.0,0.0,0.0,1.0]);
            shader.set_float4x4("V",  false, &[
                1.0,0.0,0.0,0.0,
                0.0,1.0,0.0,0.0,
                0.0,0.0,1.0,0.0,
                0.0,0.0,0.0,1.0]);
        }

        fn provide_ui(shader: &Program, win: &WindowInfo){ //shader will be already enabled
            shader.set_float4x4("P",  false, &ortho(
                                                   0.0,
                                                   win.width as f32,
                                                   win.height as f32,
                                                   0.0,
                                                   -1.0,
                                                   1.0));
            shader.set_float4x4("V",  false, &[
                1.0,0.0,0.0,0.0,
                0.0,1.0,0.0,0.0,
                0.0,0.0,1.0,0.0,
                0.0,0.0,0.0,1.0]);
        }




        match trans{
            RenderTransform::None => {
                match renderer.provider.shader_data{
                    None => renderer.provider.shader_data = Some(Box::new(provide_def)),
                    Some(x) =>{
                        let provide_def_combined = move |shader: &Program, win: &WindowInfo|{
                            provide_def(shader, win);
                            x(shader, win);
                        };

                        renderer.provider.shader_data = Some(Box::new(provide_def_combined) as Box<Fn(&Program, &WindowInfo)>);
                    }
                }
            },
            RenderTransform::UI => {
              match renderer.provider.shader_data{
                  None => renderer.provider.shader_data = Some(Box::new(provide_ui)),
                  Some(x) => {
                      let provide_ui_combined = move |shader: &Program, win: &WindowInfo|{
                          provide_ui(shader, win);
                          x(shader, win);
                      };

                      renderer.provider.shader_data = Some(Box::new(provide_ui_combined) as Box<Fn(&Program, &WindowInfo)>);
                  }
              }
            },
            RenderTransform::World => {
                unimplemented!();
            }
        };

        let rid = RenderID::ID(self.render_id_counter.get());


        match life{
            RenderLifetime::Manual => {
                self.lifetime_manual_renderers.insert(rid, renderer);
            },
            RenderLifetime::OneDraw => {
                self.lifetime_one_draw_renderers.insert(rid, renderer);
            }

        };

        self.render_id_counter.set(self.render_id_counter.get() + 1);

        Ok(rid)
    }
}