
import numpy as np

from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, no_update
import dash_bootstrap_components as dbc

import plotly.graph_objects as go

from PIL import ImageDraw, Image

def map_to_rgb(num, **rgb_params):
    
    
    
    if len(rgb_params)>0:

        normalized = (num - rgb_params['offset'])/rgb_params['range']

        
        # Map to a color between blue, gray, and red
        rgb_lower = rgb_params['rgb_lower']
        rgb_middle = rgb_params['rgb_middle']
        rgb_upper = rgb_params['rgb_upper']
    else:
    
        # Normalize to the range of 0 to 1
    
    
    
        normalized = (num + 2) / 4
        
        # Map to a color between blue, gray, and red
        rgb_lower = (0, 0, 255,1)
        rgb_middle = (128, 128, 128,1)
        rgb_upper = (255, 0, 0,1)
    
    if normalized < 0.5:
        r = int((2 * normalized) * rgb_middle[0] + (1 - 2 * normalized) * rgb_lower[0])
        g = int((2 * normalized) * rgb_middle[1] + (1 - 2 * normalized) * rgb_lower[1])
        b = int((2 * normalized) * rgb_middle[2] + (1 - 2 * normalized) * rgb_lower[2])
        a = int((2 * normalized) * rgb_middle[3] + (1 - 2 * normalized) * rgb_lower[3])
    else:
        r = int((2 * normalized - 1) * rgb_upper[0] + (2 - 2 * normalized) * rgb_middle[0])
        g = int((2 * normalized - 1) * rgb_upper[1] + (2 - 2 * normalized) * rgb_middle[1])
        b = int((2 * normalized - 1) * rgb_upper[2] + (2 - 2 * normalized) * rgb_middle[2])
        a = int((2 * normalized - 1) * rgb_upper[3] + (2 - 2 * normalized) * rgb_middle[3])
    
    # if normalized>0.5:
    #     print(normalized,a,rgb_upper[3])
    #     term
    return (r, g, b,a)

import numpy as np
from PIL import ImageDraw, Image, ImageFont

def binary_matrix_to_image(binary_matrix, row_labels=None, font_path=None, grid_size=10, border_size=1, label_size=15, is_fill=True,**rgb_params):
    # Add a dummy column to the binary matrix
    ddd=2

    height, width = binary_matrix.shape[:2]
    binary_matrix = np.concatenate((np.zeros((height, ddd)), binary_matrix), axis=1)
    width += ddd
    
    # Calculate the size of the output image based on the size of the binary matrix
    image_width = (width + 1) * grid_size + (width + 2) * border_size
    image_height = height * grid_size + (height + 1) * border_size
    
    # Create a new image and a draw object to draw the grid and borders
    image = Image.new('RGBA', (image_width, image_height), color=(0,0,0,0))
    draw = ImageDraw.Draw(image)
    
    # Draw the white grids
    for i in range(height):
        for j in range(1, width):
            # if binary_matrix[i, j] == 1:
            #     x1 = j * (grid_size + border_size) + border_size
            #     y1 = i * (grid_size + border_size) + border_size
            #     x2 = x1 + grid_size
            #     y2 = y1 + grid_size
            #     draw.rectangle((x1, y1, x2, y2), fill='white')

            if binary_matrix[i, j] != 0:
                x1 = j * (grid_size + border_size) + border_size
                y1 = i * (grid_size + border_size) + border_size
                x2 = x1 + grid_size
                y2 = y1 + grid_size

                int_color = ( int(binary_matrix[i, j]*96),
                             int(binary_matrix[i, j]*96),
                             int(binary_matrix[i, j]*96))

                
                int_color = map_to_rgb(binary_matrix[i, j], **rgb_params)
                if is_fill:
                    draw.rectangle((x1, y1, x2, y2), fill= int_color)
                else:
                    draw.rectangle((x1, y1, x2, y2), outline=int_color,width=2, fill= None)


    # Draw the borders
    color_border = (0,0,0,1)
    for i in range(height + 1):
        y = i * (grid_size + border_size)
        draw.line((grid_size, y, image_width, y), fill=color_border, width=border_size)
        
    for j in range(width + 1):
        x = j * (grid_size + border_size)
        draw.line((x, 0, x, image_height), fill=color_border, width=border_size)

    
    # Draw the row labels
    if row_labels is not None:
        font = ImageFont.truetype(PATH_SYS+'arial.ttf', size=label_size)
        max_label_width = max([font.getsize(str(label))[0] for label in row_labels])
        label_x = 0
        label_y = border_size
        for i, label in enumerate(row_labels):
            draw.text((label_x, label_y), str(label), font=font, fill='white',align="right")
            label_y += grid_size + border_size
            # if i == 0:
            #     label_x += max_label_width + border_size + grid_size
        
    return image




def run_dash_app(fig_tsne,port=None):

    if port is None:
        port = np.random.randint(2000,5000)

    app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container([
        # className="container",
        # children=[

            dbc.Row([
                    html.Div(id="dummy2"),
                    html.Div(id="dummy"),
                ]),

            dbc.Row([
                
                dbc.Col(
                    dcc.Graph(id="graph-5", figure=fig_tsne, clear_on_unhover=True,),

                    
                    width=6),  # first column with graph



                dbc.Col([
                    html.Img(id='image_merge',src='./local/images/C0-1296.jpeg',style={"height": "400px", 'display': 'block', 'margin': '0 auto'},),

                    html.Img(id='image_att',src='./local/images/C0-1296.jpeg',style={"height": "400px", 'display': 'block', 'margin': '0 auto'},),

                ], width=6),  # second column with image
            ]),

            dbc.Row([
                
                dbc.Col(
                    dcc.Graph(id="graph-summary", figure=go.Figure(), clear_on_unhover=True,),

                    
                    width=12),  # first column with graph



                # dbc.Col(
                #     html.Img(id='image',src='./local/images/C0-1296.jpeg',style={"height": "400px", 'display': 'block', 'margin': '0 auto'},),
                
                
                # width=6),  # second column with image
            ]),

            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),


        
        ])

    @app.callback(
        # Output("graph-tooltip-5", "show"),
        # Output("graph-tooltip-5", "bbox"),
        # Output("graph-tooltip-5", "children"),
        Output('image_merge', 'src'),
        Output('image_att', 'src'),
        Output("dummy2", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            # return False, no_update, no_update, no_update, no_update
            return no_update,no_update, no_update

        num = 111111
        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]


        # im_matrix = res[num].astype(int)
        # # im_url = np_image_to_base64(im_matrix)
        # # im_url = binary_matrix_to_image(im_matrix)
        # im_url = binary_matrix_to_image(im_matrix, row_labels=res_labels, grid_size=50, border_size=2, label_size=20)
        



        # # bar plot
        # vector = im_matrix.sum(1)/im_matrix.shape[1]*24
        # # im_url = plot_bar_chart(vector, labels=res_labels)
        # im_url = create_barplot_image(vector, res_labels, './local/images/temp.png')




        # NEW
        i_b = df.iloc[num]['i_b']
        i = df.iloc[num]['i']
        im_url, temp = cool_image(out,i_b,i,opt)

        
        output_str = f"{num} - {df.iloc[num]['color']} - i_b {i_b} - i {i}"
        
        im_url.save("./local/images/hover_img.png")
        im_url_path = './local/images/C2-203.jpeg'
        
        
        
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"height": "400px", 'display': 'block', 'margin': '0 auto'},
                ),
                # html.P("MNIST Digit " + str(labels[num]), style={'font-weight': 'bold'})
                html.P(f"Patterns-id={num} - i_b {i_b} - i {i}" , style={'font-weight': 'bold'})

            ])
            
        ]

        # return True, bbox, children,im_url, output_str
        return im_url,temp[0], output_str





    # Define a callback function to print the selected point IDs
    @app.callback(Output("dummy", "children"),Output("graph-summary", "figure"), [Input("graph-5", "selectedData"), Input("graph-summary", "figure")])
    def display_selected_data(selected_data, fig_prev):
        if selected_data is None:
            return "No points selected.",no_update
        else:


            new_fig = go.Figure(data=fig_prev['data'],layout=fig_prev['layout'])
            point_ids = [point["pointIndex"] for point in selected_data["points"]]

            new_fig = bar_summary(res, point_ids,res_labels, fig=new_fig)
            
            # print(f"{point_ids},\n")
            return f"{point_ids},\n",new_fig

    if __name__ == "__main__":
        # app.run_server(mode='inline', debug=True)
        app.run_server(mode='external',port=port)